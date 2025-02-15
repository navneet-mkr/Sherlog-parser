"""API endpoints for log analysis."""

import os
from pathlib import Path
from typing import Sequence, Dict, Any, cast
from fastapi import FastAPI, HTTPException
import pandas as pd

from src.models.config import Settings

app = FastAPI(title="Log Parser API")
settings = Settings()

@app.get("/templates")
async def get_templates() -> Sequence[Dict[str, Any]]:
    """Get all extracted log templates."""
    try:
        templates_file = Path(settings.output_dir) / "templates.csv"
        if not templates_file.exists():
            return []
        
        templates_df = pd.read_csv(templates_file)
        return cast(Sequence[Dict[str, Any]], templates_df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get templates: {str(e)}"
        )

@app.get("/logs/{template_id}")
async def get_logs_by_template(template_id: str) -> Sequence[Dict[str, Any]]:
    """Get all logs matching a specific template."""
    try:
        logs_file = Path(settings.output_dir) / "parsed_logs.csv"
        if not logs_file.exists():
            return []
        
        logs_df = pd.read_csv(logs_file)
        template_logs = logs_df[logs_df['template_id'] == template_id]
        return cast(Sequence[Dict[str, Any]], template_logs.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get logs: {str(e)}"
        ) 