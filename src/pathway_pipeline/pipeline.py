"""
Main pipeline implementation for log parsing using Pathway.
"""

import os
from pathlib import Path
from typing import Optional
from datetime import datetime

import pathway as pw
import pandas as pd

from .schema import LogEntrySchema, LogTemplateSchema
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer

@pw.udf
def parse_timestamp(timestamp_str: str) -> pw.DateTimeUtc:
    """Parse timestamp string to Pathway UTC datetime."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return pw.DateTimeUtc.from_python(dt)
    except (ValueError, AttributeError):
        return pw.DateTimeUtc.from_python(datetime.utcnow())

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
        
        # Add parsed timestamp
        self.logs = self.logs.select(
            content=pw.this.content,
            timestamp=parse_timestamp(pw.this.timestamp),
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
        # Process logs in batches
        processed_logs = []
        current_logs = []
        
        # Convert streaming table to list for batch processing
        for log in self.logs:
            current_logs.append(log.content)
            
            if len(current_logs) >= self.batch_size:
                results = self.log_parser.process_logs(current_logs)
                processed_logs.extend(results)
                current_logs = []
        
        # Process remaining logs
        if current_logs:
            results = self.log_parser.process_logs(current_logs)
            processed_logs.extend(results)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame({
            'log_id': range(len(processed_logs)),
            'content': [log.content for log in self.logs],
            'timestamp': [log.timestamp for log in self.logs],
            'log_level': [log.log_level for log in self.logs],
            'source': [log.source for log in self.logs],
            'template': [r.template for r in processed_logs],
            'variables': [r.variables for r in processed_logs],
            'inference_time': [r.inference_time_ms for r in processed_logs]
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
        templates = self.parsed_logs.groupby(
            template=pw.this.template,
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