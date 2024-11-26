# Company mAIstro

Company mAIstro researches information about a user-supplied list of companies, and returns it in any user-defined schema.

## Quickstart

1. Populate the `.env` file: 
```
$ cp .env.example .env
```

2. Load this folder in [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download) 

3. Provide a schema for the output, and pass in a list of companies. 

4. Run the graph!

![Screenshot 2024-11-26 at 2 33 07 PM](https://github.com/user-attachments/assets/7b52ac0a-fe4a-414c-8936-9a2d8abaea46)


## Overview

 Company mAIstro follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for better resource management and significantly reducing overall research time:

   - **Planning Phase**: An LLM analyzes the user's set of companies to research and returns a list. 
   - **Research Phase**: The system parallelizes web research across all companies in parallel:
     - Uses [Tavily API](https://tavily.com/) for targeted web searches, performing up to `max_search_queries` queries per company.
     - Performs web searches for each company in parallel and returns up to `max_search_results` results per query.
   - **Extract Schema**: After research is complete, the system uses an LLM to extract the information from the research in the user-defined schema.

## Configuration

The configuration for Company mAIstro is defined in the `configuration.py` file: 
* `max_search_queries`: int = 3 # Max search queries per company
* `max_search_results`: int = 3 # Max search results per query

These can be added in Studio:

![Screenshot 2024-11-26 at 2 33 14 PM](https://github.com/user-attachments/assets/305cf2ad-a664-4cb6-99e5-e86bd024b065)

## Inputs 

The user inputs are: 

```
* companies: List[str] - A list of companies to research
* schema: str - A JSON schema for the output
* user_notes: Optional[str] - Any additional notes about the companies from the user
```

Here is an example schema that can be supplied to research a single company:  

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
