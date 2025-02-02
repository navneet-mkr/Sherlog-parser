"""Models for regex pattern generation and validation."""

from typing import Dict, List
from pydantic import BaseModel, Field

class RegexGenerationPrompt(BaseModel):
    """Model for structuring regex generation prompts."""
    
    sample_lines: List[str]
    field_requirements: List[str] = Field(
        default=[
            "timestamp",
            "log_level",
            "message"
        ]
    )
    
    def format_prompt(self) -> str:
        """Format the prompt for the LLM."""
        lines_str = "\n".join(f"{i+1}. {line}" for i, line in enumerate(self.sample_lines))
        fields_str = "\n".join(f"- {field}" for field in self.field_requirements)
        
        return f"""Generate a Python regex pattern that matches these log lines:

{lines_str}

The pattern should:
1. Use named capture groups (e.g., (?P<timestamp>...)) for variable parts
2. Be returned as a raw string (r"...")
3. Include these required fields if present: 
{fields_str}
4. Add any additional meaningful field names that describe the captured data
5. Be as specific as possible while matching all sample lines

Respond with ONLY a JSON object containing:
{{
    "pattern": "the raw regex pattern",
    "field_descriptions": {{
        "field_name": "description of what this field captures"
    }}
}}"""

class RegexGenerationResponse(BaseModel):
    """Model for LLM regex generation responses."""
    
    pattern: str
    field_descriptions: Dict[str, str]

class PatternModification(BaseModel):
    """Model for tracking pattern modifications."""
    
    cluster_id: int
    original_pattern: str
    modified_pattern: str
    sample_line: str
    success: bool = True
    error_message: str = "" 