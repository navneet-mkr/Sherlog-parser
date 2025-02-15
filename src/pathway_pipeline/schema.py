"""
Schema definitions for the log parsing pipeline.
"""

import pathway as pw
from typing import Optional, List, Dict

class LogEntrySchema(pw.Schema):
    """Schema for raw log entries."""
    content: str
    timestamp: str
    log_level: str
    source: str

class LogTemplateSchema(pw.Schema):
    """Schema for log templates."""
    template_id: str
    template: str
    parameters: List[str]
    description: Optional[str]

class ParsedLogSchema(pw.Schema):
    """Schema for parsed log entries."""
    content: str
    timestamp: pw.DateTimeUtc
    log_level: str
    source: str
    template_id: str
    parsed_parameters: Dict[str, str]
    event_type: str
    severity: Optional[str] 