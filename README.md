# Company mAIstro

Company mAIstro researches information about a user-supplied company, and returns it in any user-defined schema.

## Quickstart

1. Populate the `.env` file: 
```
$ cp .env.example .env
```

2. Load this folder in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) 

3. Provide a schema for the output, and pass in a company name. 

4. Run the graph!

![Screenshot 2024-11-26 at 8 14 15 PM](https://github.com/user-attachments/assets/f2a9724b-12a9-41d8-bf1f-21c2217ba400)


## Overview

Company mAIstro follows a multi-step research and extraction workflow that separates web research from schema extraction, allowing for better resource management and comprehensive data collection:

   - **Research Phase**: The system performs intelligent web research on the input company:
     - Uses an LLM to generate targeted search queries based on the schema requirements (up to `max_search_queries` per company)
     - Executes concurrent web searches via [Tavily API](https://tavily.com/), retrieving up to `max_search_results` results per query
     - Takes structured research notes focused on schema-relevant information
   - **Extraction Phase**: After research is complete, the system:
     - Consolidates all research notes
     - Uses an LLM to extract and format the information according to the user-defined schema
     - Returns the structured data in the exact format requested

## Configuration

The configuration for Company mAIstro is defined in the `configuration.py` file: 
* `max_search_queries`: int = 3 # Max search queries per company
* `max_search_results`: int = 3 # Max search results per query

These can be added in Studio:

![Screenshot 2024-11-26 at 8 14 19 PM](https://github.com/user-attachments/assets/2c102a80-692d-479d-a5a7-edd6506eb42d)

## Inputs 

The user inputs are: 

```
* company: str - A company to research
* schema: dict - A JSON schema for the output
* user_notes: Optional[str] - Any additional notes about the company from the user
```

Here is an example schema that can be supplied to research a company:  

> ⚠️ **WARNING:** JSON schemas require `title` and `description` fields for [extraction](https://python.langchain.com/docs/how_to/structured_output/#typeddict-or-json-schema). Otherwise, you may see errors as shown [here](https://smith.langchain.com/public/341dba26-cff8-447b-b940-9f097d43bfa2/r).

```
{
    "title": "CompanyInfo",
    "description": "Basic information about a company",
    "type": "object",
    "properties": {
        "company_name": {
            "type": "string",
            "description": "Official name of the company"
        },
        "founding_year": {
            "type": "integer",
            "description": "Year the company was founded"
        },
        "founder_names": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Names of the founding team members"
        },
        "product_description": {
            "type": "string",
            "description": "Brief description of the company's main product or service"
        },
        "funding_summary": {
            "type": "string",
            "description": "Summary of the company's funding history"
        }
    },
    "required": ["company_name"]
}
```

## Evaluation

### Datasets

- [Public companies](https://smith.langchain.com/public/bb139cd5-c656-4323-9bea-84cb7bf6080a/d)
- [Startups](https://smith.langchain.com/public/2b0a2f35-9d7c-40f2-a24f-5dec877dec1e/d)

### Metric

Currently there is a single evaluation metric: fraction of the fields that were correctly extracted (per company). Correctness is defined differently depending on the field type:

- exact matches for fields like `founding_year` / `website`
- fuzzy matches for fields like `company_name` / `ceo`
- embedding similarity for fields like `description`
- checking within a certain tolerance (+/- 10%) for fields like `employee_count` / `total_funding_mm_usd`

### Running evals

To evaluate the Company mAIstro agent, you can run `evals/test_agent.py` script. This will create new experiments in LangSmith for the two [datasets](#datasets) mentioned above.

Basic usage:

```shell
python evals/test_agent.py
```

By default the script will also check the results against the minimum acceptable scores. This can be skipped by passing in the `--skip-regression` flag.

```shell
python evals/test_agent.py --skip-regression
```

You can also customize additional parameters such as the maximum number of concurrent runs and the experiment prefix.

```shell
python evals/test_agent.py --max-concurrency 4 --experiment-prefix "My custom prefix"
```