"""
Main pipeline implementation for log parsing using Pathway.
"""

import os
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, UTC
from dataclasses import dataclass

import pathway as pw
import pandas as pd

from .schema import LogEntrySchema, LogTemplateSchema, ParsedLogSchema
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer

@dataclass
class PipelineConfig:
    """Configuration for the log parsing pipeline."""
    input_dir: str
    output_dir: str
    cache_dir: str
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "ollama/mistral"

@pw.udf
def parse_timestamp(timestamp_str: Union[str, int, float]) -> pw.DateTimeUtc:
    """Parse timestamp string to Pathway UTC datetime.
    
    Handles multiple timestamp formats:
    - ISO format with timezone (e.g., '2023-05-15T10:13:00+01:00')
    - ISO format without timezone (assumes UTC, e.g., '2023-05-15T10:13:00')
    - Unix timestamp in milliseconds
    """
    try:
        # Try parsing as Unix timestamp (milliseconds)
        if isinstance(timestamp_str, (int, float)):
            return pw.DATE_TIME_UTC.from_timestamp(timestamp_str, unit="ms")
            
        # Try parsing as ISO format
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str.replace('Z', '+00:00')
        if '+' in timestamp_str or '-' in timestamp_str:
            # Has timezone info
            dt = datetime.fromisoformat(timestamp_str)
            return pw.DateTimeUtc(dt)
        else:
            # No timezone - assume UTC
            dt = datetime.fromisoformat(timestamp_str)
            dt = dt.replace(tzinfo=UTC)
            return pw.DateTimeUtc(dt)
    except (ValueError, AttributeError) as e:
        print(f"Error parsing timestamp {timestamp_str}: {e}")
        return pw.DateTimeUtc(datetime.now(UTC))

class LogParsingPipeline:
    """Main pipeline for log parsing using Pathway."""
    
    def __init__(
        self,
        log_dir: str,
        template_file: Optional[str] = None,
        llm_model: str = "ollama/mistral",
        llm_api_base: str = "http://localhost:11434",
        output_dir: str = "./output",
        cache_dir: str = "./cache",
        similarity_threshold: float = 0.8,
        batch_size: int = 32
    ):
        self.log_dir = Path(log_dir)
        self.template_file = template_file
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Initialize Ollama analyzer
        self.ollama_analyzer = create_ollama_analyzer(
            base_url=llm_api_base,
            model_id=llm_model.split('/')[-1],
            config={"temperature": 0.1}
        )
        
        # Initialize log parser
        self.log_parser = LogParserLLM(
            ollama_client=self.ollama_analyzer,
            similarity_threshold=similarity_threshold
        )
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_pipeline(self) -> None:
        """Set up the Pathway pipeline for log parsing."""
        # Read log files
        self.logs = pw.io.fs.read(
            os.path.join(self.log_dir, "*.log"),
            format="text",
            schema=LogEntrySchema,
            mode="streaming"
        )
        
        # Add parsed timestamp while preserving original
        self.logs = self.logs.select(
            content=pw.this.content,
            timestamp=pw.this.timestamp,  # Keep original string
            parsed_timestamp=parse_timestamp(pw.this.timestamp),  # Add parsed UTC timestamp
            log_level=pw.this.log_level,
            source=pw.this.source
        )
        
        # Read existing templates if provided
        if self.template_file:
            self.templates = pw.io.csv.read(
                self.template_file,
                schema=LogTemplateSchema,
                mode="streaming"
            )
        
        # Process logs
        self.parsed_logs = self._process_logs()
        
        # Set up outputs
        self._setup_outputs()
    
    def _process_logs(self) -> pw.Table:
        """Process logs using LogParserLLM."""
        # Convert Pathway table to list for batch processing
        temp_csv = os.path.join(self.cache_dir, "temp_logs.csv")
        pw.io.csv.write(self.logs, temp_csv)
        log_contents = pd.read_csv(temp_csv)
        os.remove(temp_csv)  # Clean up
        
        # Process logs in batches
        processed_logs = []
        
        for idx, row in enumerate(log_contents.itertuples()):
            result = self.log_parser.parse_log(str(row[1]), idx)  # content is at index 1
            processed_logs.append(result)
            
        # Convert results to DataFrame
        results_df = pd.DataFrame({
            'log_id': range(len(processed_logs)),
            'content': log_contents['content'].tolist(),
            'timestamp': log_contents['timestamp'].tolist(),
            'log_level': log_contents['log_level'].tolist(),
            'source': log_contents['source'].tolist(),
            'template': [r[0] for r in processed_logs],  # template is first element of tuple
            'variables': [r[1] for r in processed_logs],  # parameters dict is second element
            'inference_time': [0 for _ in processed_logs]  # No inference time in parse_log
        })
        
        return pw.debug.table_from_pandas(results_df)
    
    def _setup_outputs(self) -> None:
        """Set up output connectors."""
        # Write parsed logs to CSV
        pw.io.csv.write(
            self.parsed_logs,
            os.path.join(self.output_dir, "parsed_logs.csv")
        )
        
        # Write unique templates to CSV
        templates = self.parsed_logs.groupby(pw.this.template).reduce(
            count=pw.reducers.count()
        )
        pw.io.csv.write(
            templates,
            os.path.join(self.output_dir, "templates.csv")
        )
    
    def run(self) -> None:
        """Run the pipeline."""
        pw.run()

def main():
    """Main entry point."""
    pipeline = LogParsingPipeline(
        log_dir="./data/logs",
        output_dir="./output",
        cache_dir="./cache"
    )
    pipeline.setup_pipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 