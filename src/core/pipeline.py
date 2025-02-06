"""Module for handling the complete log processing pipeline."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from dagster import (
    job, op, In, Out, Nothing,
    DagsterType, TypeCheck,
    Field as DagsterField, String, Int, Bool,
    AssetMaterialization, MetadataValue,
    ResourceDefinition
)

from src.models.log_parser import LogParserLLM
from src.models.storage import DuckDBStorage, ParsedLog
from src.models.ollama import create_ollama_analyzer

logger = logging.getLogger(__name__)

# Custom Dagster types
def is_valid_log_lines(_, value: List[str]) -> TypeCheck:
    """Validate log lines."""
    if not isinstance(value, list):
        return TypeCheck(
            success=False,
            description="Value should be a list of strings"
        )
    if not all(isinstance(line, str) for line in value):
        return TypeCheck(
            success=False,
            description="All elements should be strings"
        )
    return TypeCheck(success=True)

LogLines = DagsterType(
    name="LogLines",
    type_check_fn=is_valid_log_lines,
    description="A list of log lines"
)

# Pipeline operations
@op(
    ins={"file_path": In(String)},
    out=Out(LogLines),
    config_schema={
        "encoding": DagsterField(String, default_value="utf-8", is_required=False)
    }
)
def read_log_file(context, file_path: str) -> List[str]:
    """Read log lines from file."""
    try:
        encoding = context.op_config["encoding"]
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        lines = path.read_text(encoding=encoding).splitlines()
        
        context.log.info(f"Read {len(lines)} lines from {file_path}")
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key=f"log_file_{path.stem}",
                description=f"Log file: {file_path}",
                metadata={
                    "num_lines": MetadataValue.int(len(lines)),
                    "file_size": MetadataValue.int(path.stat().st_size)
                }
            )
        )
        
        return lines
    except Exception as e:
        context.log.error(f"Failed to read file: {str(e)}")
        raise

@op(
    ins={"log_lines": In(LogLines)},
    out=Out(dict),
    config_schema={
        "ollama_base_url": DagsterField(String, default_value="http://localhost:11434"),
        "model_name": DagsterField(String, default_value="mistral"),
        "similarity_threshold": DagsterField(float, default_value=0.8),
        "batch_size": DagsterField(Int, default_value=1000)
    }
)
def parse_logs_llm(context, log_lines: List[str]) -> Dict[str, Any]:
    """Parse logs using LogParser-LLM algorithm."""
    try:
        # Create Ollama client
        ollama_client = create_ollama_analyzer(
            base_url=context.op_config["ollama_base_url"],
            model_id=context.op_config["model_name"],
            config={}
        )
        
        # Initialize LogParser-LLM
        parser = LogParserLLM(
            ollama_client=ollama_client,
            similarity_threshold=context.op_config["similarity_threshold"]
        )
        
        # Process logs in batches
        batch_size = context.op_config["batch_size"]
        parsed_logs = []
        
        for i in range(0, len(log_lines), batch_size):
            batch = log_lines[i:i + batch_size]
            for log_id, log_message in enumerate(batch, start=i):
                template, parameters = parser.parse_log(log_message, log_id)
                parsed_logs.append(
                    ParsedLog(
                        log_id=log_id,
                        raw_log=log_message,
                        log_template=template,
                        parameters=parameters
                    )
                )
            
            context.log.info(f"Processed batch {i//batch_size + 1}")
        
        # Get parser statistics
        stats = parser.get_statistics()
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key="parsed_logs",
                description="Parsed log data",
                metadata={
                    "total_logs": MetadataValue.int(len(parsed_logs)),
                    "total_templates": MetadataValue.int(stats["total_templates"]),
                    "total_clusters": MetadataValue.int(stats["total_clusters"])
                }
            )
        )
        
        return {
            "parsed_logs": parsed_logs,
            "statistics": stats
        }
    except Exception as e:
        context.log.error(f"Failed to parse logs: {str(e)}")
        raise

@op(
    ins={"parsed_data": In(dict)},
    out=Out(Nothing),
    config_schema={
        "db_path": DagsterField(String, default_value=":memory:", is_required=False),
        "persist_db": DagsterField(Bool, default_value=True, is_required=False)
    }
)
def store_parsed_logs(context, parsed_data: Dict[str, Any]):
    """Store parsed logs in DuckDB."""
    try:
        db_path = context.op_config["db_path"]
        if not context.op_config["persist_db"]:
            db_path = ":memory:"
            
        # Initialize DuckDB storage
        storage = DuckDBStorage(db_path=db_path)
        
        # Store parsed logs
        parsed_logs = parsed_data["parsed_logs"]
        storage.insert_parsed_logs(parsed_logs)
        
        # Get and log template statistics
        template_stats = storage.get_template_statistics()
        
        context.log.info(
            f"Stored {len(parsed_logs)} logs with "
            f"{len(template_stats)} unique templates"
        )
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key="duckdb_storage",
                description=f"DuckDB storage at {db_path}",
                metadata={
                    "total_logs": MetadataValue.int(len(parsed_logs)),
                    "unique_templates": MetadataValue.int(len(template_stats)),
                    "persistent": MetadataValue.bool(context.op_config["persist_db"])
                }
            )
        )
        
        storage.close()
    except Exception as e:
        context.log.error(f"Failed to store parsed logs: {str(e)}")
        raise

@job
def log_parsing_pipeline():
    """Main log parsing pipeline using LogParser-LLM."""
    # Read logs
    log_lines = read_log_file()
    
    # Parse logs using LogParser-LLM
    parsed_data = parse_logs_llm(log_lines)
    
    # Store results in DuckDB
    store_parsed_logs(parsed_data) 