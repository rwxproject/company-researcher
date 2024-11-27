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

![Screenshot 2024-11-26 at 7 58 52 PM](https://github.com/user-attachments/assets/b12857f4-0810-43c2-805c-d85e7a235034)

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

![Screenshot 2024-11-26 at 7 56 03 PM](https://github.com/user-attachments/assets/db12c1cf-34bb-4773-86a5-a1e1c3519be9)


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
