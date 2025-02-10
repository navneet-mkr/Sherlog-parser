"""
Main pipeline implementation for log parsing using Pathway.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

import pathway as pw
from pathway.stdlib.indexing import default_vector_document_index
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.llms import LiteLLMChat

from .schema import LogEntrySchema, LogTemplateSchema, ParsedLogSchema

@pw.udf
def build_template_prompt(content: str) -> str:
    """Build prompt for template extraction."""
    return f"""Extract a template and variables from this log message:
{content}

The template should replace variable parts with placeholders like <timestamp>, <ip>, <number>, etc.
Return a JSON object with:
- template: the extracted template
- variables: list of variable positions and types

Example:
Log: "2024-02-07 10:15:30 ERROR Connection failed from 192.168.1.100"
{{"template": "<timestamp> ERROR Connection failed from <ip>",
  "variables": [
    {{"position": 0, "type": "timestamp"}},
    {{"position": 4, "type": "ip"}}
  ]
}}"""

@pw.udf
def parse_llm_response(response: str) -> dict:
    """Parse LLM response into template and variables."""
    try:
        result = json.loads(response)
        return {
            "template": result["template"],
            "variables": {
                f"var_{v['position']}": v["type"]
                for v in result["variables"]
            }
        }
    except (json.JSONDecodeError, KeyError):
        return {"template": "", "variables": {}}

@pw.udf
def extract_parameters(content: str, template: str, variables: dict) -> dict:
    """Extract parameters from log message using template."""
    log_tokens = content.split()
    template_tokens = template.split()
    
    parameters = {}
    
    if len(log_tokens) != len(template_tokens):
        return parameters
    
    for i, (log_token, template_token) in enumerate(zip(log_tokens, template_tokens)):
        if template_token.startswith("<") and template_token.endswith(">"):
            var_name = f"param_{i}"
            parameters[var_name] = log_token
    
    return parameters

class LogParsingPipeline:
    def __init__(
        self,
        log_dir: str,
        template_file: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "ollama/mistral",
        llm_api_base: str = "http://localhost:11434",
        output_dir: str = "./output",
    ):
        self.log_dir = log_dir
        self.template_file = template_file
        self.output_dir = output_dir
        
        # Initialize embedder
        self.embedder = embedders.SentenceTransformerEmbedder(
            embedding_model,
            call_kwargs={"show_progress_bar": False}
        )
        
        # Initialize LLM
        self.model = LiteLLMChat(
            model=llm_model,
            temperature=0,
            api_base=llm_api_base,
            format="json"
        )
        
    def setup_pipeline(self) -> None:
        """Set up the Pathway pipeline for log parsing."""
        # Read log files
        self.logs = pw.io.fs.read(
            os.path.join(self.log_dir, "*.log"),
            format="text",
            schema=LogEntrySchema,
            mode="streaming"
        )
        
        # Read templates
        self.templates = pw.io.csv.read(
            self.template_file,
            schema=LogTemplateSchema,
            mode="streaming"
        )
        
        # Create template index
        self.template_index = default_vector_document_index(
            self.templates.template,
            self.templates,
            embedder=self.embedder,
            dimensions=self.embedder.get_embedding_dimension()
        )
        
        # Parse logs
        self.parsed_logs = self._parse_logs()
        
        # Set up outputs
        self._setup_outputs()
    
    def _parse_logs(self) -> pw.Table:
        """Parse logs using LLM with template matching."""
        # First attempt to match against existing templates
        logs_with_matches = self.logs.join(
            self.template_index.get_nearest_items(
                self.logs.content,
                k=3,
                distance_threshold=0.2
            ),
            pw.left.content == pw.right.query
        )
        
        # For logs without matches, use LLM to extract templates
        logs_without_matches = (
            self.logs
            .filter(lambda t: t.id not in logs_with_matches.id)
            .select(
                prompt=build_template_prompt(pw.this.content)
            )
            .select(
                llm_response=self.model(pw.this.prompt)
            )
            .select(
                parsed_response=parse_llm_response(pw.this.llm_response)
            )
        )
        
        # Extract parameters for matched logs
        matched_logs = logs_with_matches.select(
            content=pw.this.content,
            timestamp=pw.this.timestamp,
            log_level=pw.this.log_level,
            source=pw.this.source,
            template_id=pw.this.template_id,
            parsed_parameters=extract_parameters(
                pw.this.content,
                pw.this.template,
                pw.this.parameters
            ),
            event_type=pw.this.description,
            severity=pw.this.log_level
        )
        
        # Process newly extracted templates
        new_templates = logs_without_matches.select(
            content=pw.this.content,
            timestamp=pw.this.timestamp,
            log_level=pw.this.log_level,
            source=pw.this.source,
            template_id=f"new_{pw.this.id}",
            parsed_parameters=extract_parameters(
                pw.this.content,
                pw.this.parsed_response["template"],
                pw.this.parsed_response["variables"]
            ),
            event_type="unknown",
            severity=pw.this.log_level
        )
        
        # Combine results
        return pw.Table.concat(matched_logs, new_templates)
    
    def _setup_outputs(self) -> None:
        """Set up output connectors."""
        # Write to CSV
        pw.io.csv.write(
            self.parsed_logs,
            os.path.join(self.output_dir, "parsed_logs.csv")
        )
        
        # TODO: Add more output connectors as needed
        # (e.g., PostgreSQL, Prometheus metrics, etc.)
    
    def run(self) -> None:
        """Run the pipeline."""
        pw.run()

def main():
    """Main entry point."""
    pipeline = LogParsingPipeline(
        log_dir="./data/logs",
        template_file="./data/templates.csv",
        output_dir="./output"
    )
    pipeline.setup_pipeline()
    pipeline.run()

if __name__ == "__main__":
    main() 