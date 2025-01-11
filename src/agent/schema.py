from typing import List, Optional
from pydantic import BaseModel, Field

class ReportSource(BaseModel):
    source: str = Field(default="", description="Brief description about the source of the report")
    url: str = Field(default="", description="URL used in the source of the report")

class CompanyInfo(BaseModel):
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(description="Names of the founding team members")
    product_description: Optional[str] = Field(description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(description="Summary of the company's funding history")
    sources: Optional[List[ReportSource]] = Field(description="List of sources used to generate the report")

    model_config = {
        "json_schema_extra": {
            "title": "CompanyInfo",
            "description": "Basic information about a company"
        }
    }

DEFAULT_EXTRACTION_SCHEMA = CompanyInfo.model_json_schema(by_alias=True)