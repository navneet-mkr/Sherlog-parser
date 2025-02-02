"""FastAPI service for real-time log parsing."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from src.models import Settings, LogLine
from src.core.storage import DuckDBManager
from src.services.parser.parser import LogParser

app = FastAPI(
    title="Log Parse AI",
    description="Real-time log parsing service using ML and LLMs",
    version="1.0.0"
)

# Global instances
settings = Settings()
db = DuckDBManager(settings)
parser = LogParser(settings, db)

class ParseRequest(BaseModel):
    """Request model for log parsing."""
    log_line: str

class ParseResponse(BaseModel):
    """Response model for parsed log line."""
    raw_text: str
    cluster_id: Optional[int]
    parsed_fields: dict
    success: bool
    error: Optional[str] = None

class BatchParseRequest(BaseModel):
    """Request model for batch log parsing."""
    log_lines: List[str]

class BatchParseResponse(BaseModel):
    """Response model for batch parsing."""
    results: List[ParseResponse]
    success_count: int
    failure_count: int

@app.post("/parse", response_model=ParseResponse)
async def parse_log(request: ParseRequest) -> ParseResponse:
    """Parse a single log line."""
    try:
        result = parser.parse_line(request.log_line)
        return ParseResponse(
            raw_text=result.raw_text,
            cluster_id=result.cluster_id,
            parsed_fields=result.parsed_fields,
            success=bool(result.parsed_fields)
        )
    except Exception as e:
        return ParseResponse(
            raw_text=request.log_line,
            cluster_id=None,
            parsed_fields={},
            success=False,
            error=str(e)
        )

@app.post("/parse/batch", response_model=BatchParseResponse)
async def parse_logs(request: BatchParseRequest) -> BatchParseResponse:
    """Parse multiple log lines in batch."""
    results = []
    success_count = 0
    
    for line in request.log_lines:
        try:
            result = parser.parse_line(line)
            success = bool(result.parsed_fields)
            if success:
                success_count += 1
                
            results.append(ParseResponse(
                raw_text=result.raw_text,
                cluster_id=result.cluster_id,
                parsed_fields=result.parsed_fields,
                success=success
            ))
        except Exception as e:
            results.append(ParseResponse(
                raw_text=line,
                cluster_id=None,
                parsed_fields={},
                success=False,
                error=str(e)
            ))
    
    return BatchParseResponse(
        results=results,
        success_count=success_count,
        failure_count=len(request.log_lines) - success_count
    )

@app.get("/health")
async def health_check() -> dict:
    """Check service health."""
    return {
        "status": "healthy",
        "components": {
            "database": db.is_healthy(),
            "parser": parser is not None
        }
    } 