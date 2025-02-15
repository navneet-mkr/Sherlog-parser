"""Pydantic models for prompt responses."""

from typing import List, Optional
from pydantic import BaseModel, Field

class TemplateExtractionResponse(BaseModel):
    """Response model for template extraction."""
    template: str = Field(description="The extracted template with variables replaced by category tokens")
    variables: List[dict] = Field(
        default_factory=list,
        description="List of variables with their values, categories and positions"
    )

class MergeVerificationResponse(BaseModel):
    """Response model for merge verification."""
    answer: str = Field(pattern="^(yes|no)$", description="Whether the template applies to all logs")

class ParseResult(BaseModel):
    """Model for individual log parse results."""
    template: str
    variables: List[dict]

class MergeCheckResponse(BaseModel):
    """Response model for merge checking."""
    parse_results: List[ParseResult] = Field(default_factory=list)
    reason: str
    answer: str = Field(pattern="^(Yes|No)$")
    unified_template: Optional[str] = None 