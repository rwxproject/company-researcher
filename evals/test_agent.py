from typing import Any

from Levenshtein import ratio
from langsmith import Client, evaluate
from langsmith.evaluation import LangChainStringEvaluator

from langgraph.pregel.remote import RemoteGraph


client = Client()

# Load datasets
public_companies_dataset = client.read_dataset(
    dataset_name="Public Company Data Enrichment"
)
startup_dataset = client.read_dataset(dataset_name="Startup Data Enrichment")


TOLERANCE = 0.10  # should match within 10%
NUMERIC_FIELDS = (
    "employee_count",
    "total_funding_mm_usd",
    "latest_round_amount_mm_usd",
)
EXACT_MATCH_FIELDS = (
    "website",
    "crunchbase_profile",
    "linkedin_profile",
    "headquarters",
    "year_founded",
    "latest_round",
    "latest_round_date",
)
FUZZY_MATCH_FIELDS = ("name", "ceo")
LONG_TEXT_FIELDS = ("description",)
EXPERIMENT_PREFIX = "Company mAIstro "


# evaluation helpers for different types of fields


def evaluate_numeric_fields(outputs: dict, reference_outputs: dict) -> dict[str, float]:
    lower_bound = 1 - TOLERANCE
    upper_bound = 1 + TOLERANCE
    field_to_score = {}
    for k in NUMERIC_FIELDS:
        if k not in reference_outputs:
            continue

        raw_field_value = outputs.get(k, 0)
        try:
            score = float(
                lower_bound
                <= int(raw_field_value) / reference_outputs[k]
                <= upper_bound
            )
        except ValueError:
            score = 0.0

        field_to_score[k] = score
    return field_to_score


def _preprocess_value(value: Any) -> Any:
    if isinstance(value, str):
        # for urls
        return value.rstrip("/")

    return value


def evaluate_exact_match_fields(
    outputs: dict, reference_outputs: dict
) -> dict[str, float]:
    return {
        k: float(
            _preprocess_value(outputs.get(k)) == _preprocess_value(reference_outputs[k])
        )
        for k in EXACT_MATCH_FIELDS
        if k in reference_outputs
    }


def evaluate_long_text_fields(outputs: dict, reference_outputs: dict):
    emb_distance_evaluator = LangChainStringEvaluator(
        "embedding_distance", config={"distance": "cosine"}
    )
    return {
        k: 1
        - emb_distance_evaluator.evaluator.invoke(
            {"prediction": outputs.get(k, ""), "reference": reference_outputs[k]}
        )["score"]
        for k in LONG_TEXT_FIELDS
        if k in reference_outputs
    }


def evaluate_fuzzy_match_fields(outputs: dict, reference_outputs: dict):
    return {
        k: ratio(outputs.get(k, "").lower(), reference_outputs[k].lower())
        for k in FUZZY_MATCH_FIELDS
        if k in reference_outputs
    }


# effectively fraction of matching fields
def evaluate_agent(outputs: dict, reference_outputs: dict):
    if "info" not in outputs or not isinstance(outputs["info"], dict):
        return 0.0

    actual_company_info = outputs["info"]
    expected_company_info = reference_outputs["info"]

    results = {
        **evaluate_numeric_fields(actual_company_info, expected_company_info),
        **evaluate_exact_match_fields(actual_company_info, expected_company_info),
        **evaluate_fuzzy_match_fields(actual_company_info, expected_company_info),
    }
    return sum(results.values()) / len(results)


agent_graph = RemoteGraph(
    "company_maistro",
    url="https://company-maistro-6e9e7797491257379f0d09894f301c04.default.us.langgraph.app",
)


def run_agent(inputs: dict):
    response = agent_graph.invoke(inputs)
    return {"info": response["info"]}


MIN_PUBLIC_COMPANY_SCORE = 0.9
MIN_STARTUP_SCORE = 0.7


def run_evals(max_concurrency: int = 2, check_regression: bool = True):
    public_company_eval_results = evaluate(
        run_agent,
        data=public_companies_dataset,
        evaluators=[evaluate_agent],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=max_concurrency,
    )
    startup_eval_results = evaluate(
        run_agent,
        data=startup_dataset,
        evaluators=[evaluate_agent],
        experiment_prefix=EXPERIMENT_PREFIX,
        max_concurrency=max_concurrency,
    )

    if check_regression:
        public_company_eval_results_df = public_company_eval_results.to_pandas()
        startup_eval_results_df = startup_eval_results.to_pandas()

        public_company_score = public_company_eval_results_df[
            "feedback.evaluate_agent"
        ].mean()
        startup_score = startup_eval_results_df["feedback.evaluate_agent"].mean()

        error_msg = ""
        if public_company_score < MIN_PUBLIC_COMPANY_SCORE:
            error_msg += f"Public company score {public_company_score} is less than {MIN_PUBLIC_COMPANY_SCORE}\n"

        if startup_score < MIN_STARTUP_SCORE:
            error_msg += f"Startup score {startup_score} is less than {MIN_STARTUP_SCORE}\n"

        if error_msg:
            raise AssertionError(error_msg)

    return public_company_eval_results, startup_eval_results


if __name__ == "__main__":
    run_evals()
