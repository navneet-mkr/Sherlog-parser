"""Core pipeline implementation for log parsing using Pathway."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import pathway as pw
from pathway.stdlib.indexing import default_vector_document_index
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.llms import LiteLLMChat

from pydantic import BaseModel

class PipelineConfig(BaseModel):
    """Pipeline configuration."""
    input_dir: str
    output_dir: str
    cache_dir: str = "./cache"
    encoding: str = "utf-8"
    
    # LLM settings
    ollama_base_url: str = "http://localhost:11434"
    model_name: str = "mistral"
    similarity_threshold: float = 0.8
    batch_size: int = 1000
    
    # Database settings
    db_path: str = ":memory:"
    persist_db: bool = True

class LogParsingPipeline:
    """Main pipeline for log parsing using Pathway."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Create directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)
        
        # Initialize embedder
        self.embedder = embedders.SentenceTransformerEmbedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            call_kwargs={"show_progress_bar": False}
        )
        
        # Initialize LLM
        self.model = LiteLLMChat(
            model=config.model_name,
            temperature=0,
            api_base=config.ollama_base_url,
            format="json"
        )
    
    @pw.udf
    def _build_template_prompt(self, content: str) -> str:
        """Build prompt for template extraction."""
        return f"""Extract a template and variables from this log message:
{content}

The template should replace variable parts with placeholders.
Return a JSON object with:
- template: the extracted template with <type> placeholders
- variables: list of variable positions and types

Example:
Log: "2024-02-07 10:15:30 ERROR Connection failed from 192.168.1.100"
{{"template": "<timestamp> ERROR Connection failed from <ip>",
  "variables": [
    {{"position": 0, "type": "timestamp"}},
    {{"position": 4, "type": "ip"}}
  ]
}}"""
    
    def setup_pipeline(self) -> None:
        """Set up the Pathway pipeline."""
        # Read log files
        self.logs = pw.io.fs.read(
            os.path.join(self.config.input_dir, "*.log"),
            format="text",
            mode="streaming"
        )
        
        # Create template index from existing templates if available
        template_file = os.path.join(self.config.cache_dir, "templates.csv")
        if os.path.exists(template_file):
            self.templates = pw.io.csv.read(template_file)
            self.template_index = default_vector_document_index(
                self.templates.template,
                self.templates,
                embedder=self.embedder,
                dimensions=self.embedder.get_embedding_dimension()
            )
        else:
            self.templates = None
            self.template_index = None
        
        # Process logs
        self.parsed_logs = self._process_logs()
        
        # Set up outputs
        self._setup_outputs()
    
    def _process_logs(self) -> pw.Table:
        """Process logs and extract templates."""
        if self.template_index:
            # Try template matching first
            logs_with_matches = self.logs.join(
                self.template_index.get_nearest_items(
                    self.logs.content,
                    k=1,
                    distance_threshold=self.config.similarity_threshold
                ),
                pw.left.content == pw.right.query
            )
            
            # For unmatched logs, use LLM
            logs_without_matches = (
                self.logs
                .filter(lambda t: t.id not in logs_with_matches.id)
                .select(
                    content=pw.this.content,
                    prompt=self._build_template_prompt(pw.this.content)
                )
                .select(
                    pw.this.content,
                    llm_response=self.model(pw.this.prompt)
                )
            )
            
            # Combine results
            return pw.Table.concat(
                logs_with_matches.select(
                    content=pw.this.content,
                    template=pw.this.template,
                    template_id=pw.this.template_id
                ),
                logs_without_matches.select(
                    content=pw.this.content,
                    template=pw.apply(
                        lambda x: json.loads(x)["template"],
                        pw.this.llm_response
                    ),
                    template_id=f"new_{pw.this.id}"
                )
            )
        else:
            # Process all logs with LLM
            return (
                self.logs
                .select(
                    content=pw.this.content,
                    prompt=self._build_template_prompt(pw.this.content)
                )
                .select(
                    pw.this.content,
                    llm_response=self.model(pw.this.prompt)
                )
                .select(
                    content=pw.this.content,
                    template=pw.apply(
                        lambda x: json.loads(x)["template"],
                        pw.this.llm_response
                    ),
                    template_id=f"new_{pw.this.id}"
                )
            )
    
    def _setup_outputs(self) -> None:
        """Set up output connectors."""
        # Write to CSV
        pw.io.csv.write(
            self.parsed_logs,
            os.path.join(self.config.output_dir, "parsed_logs.csv")
        )
        
        # Save templates
        pw.io.csv.write(
            self.parsed_logs.select(
                template=pw.this.template,
                template_id=pw.this.template_id
            ).groupby(pw.this.template_id),
            os.path.join(self.config.cache_dir, "templates.csv")
        )
    
    def run(self) -> None:
        """Run the pipeline."""
        pw.run()

def main():
    """Main entry point."""
    config = PipelineConfig(
        input_dir="./data/logs",
        output_dir="./output",
        cache_dir="./cache"
    )
    
    pipeline = LogParsingPipeline(config)
    pipeline.setup_pipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 