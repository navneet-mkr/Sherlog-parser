"""
Evaluation pipeline implementation using Pathway.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

import pathway as pw
from pathway.stdlib.indexing import default_vector_document_index
from pathway.xpacks.llm import embedders
from pathway.xpacks.llm.llms import LiteLLMChat

from src.eval.datasets import DatasetLoader, LogDataset
from src.eval.metrics import evaluate_parser_output, EvaluationMetrics
from src.models.ollama import VariableType

class EvaluationPipeline:
    def __init__(
        self,
        base_dir: str,
        dataset_type: str,
        system: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "ollama/mistral",
        llm_api_base: str = "http://localhost:11434",
        output_dir: str = "./output",
        cache_dir: str = "./cache",
        similarity_threshold: float = 0.8,
        batch_size: int = 32
    ):
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.system = system
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
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
        
        # Load dataset
        self.dataset = self._load_dataset()
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_dataset(self) -> LogDataset:
        """Load dataset for evaluation."""
        loader = DatasetLoader(self.base_dir)
        return loader.load_dataset(
            system=self.system,
            dataset_type=self.dataset_type
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
        """Set up the Pathway evaluation pipeline."""
        # Convert dataset to Pathway table
        df = pd.DataFrame({
            'log_id': range(len(self.dataset.raw_logs)),
            'content': self.dataset.raw_logs,
            'ground_truth': [
                self.dataset.ground_truth_templates[i]
                for i in range(len(self.dataset.raw_logs))
            ]
        })
        self.logs = pw.debug.table_from_pandas(df)
        
        # Create template index from ground truth
        unique_templates = list(set(self.dataset.ground_truth_templates.values()))
        template_df = pd.DataFrame({
            'template_id': range(len(unique_templates)),
            'template': unique_templates
        })
        self.templates = pw.debug.table_from_pandas(template_df)
        
        self.template_index = default_vector_document_index(
            self.templates.template,
            self.templates,
            embedder=self.embedder,
            dimensions=self.embedder.get_embedding_dimension()
        )
        
        # Process logs
        self.results = self._process_logs()
        
    def _process_logs(self) -> pw.Table:
        """Process logs and generate templates."""
        # First try template matching
        logs_with_matches = self.logs.join(
            self.template_index.get_nearest_items(
                self.logs.content,
                k=1,
                distance_threshold=self.similarity_threshold
            ),
            pw.left.content == pw.right.query
        )
        
        # For unmatched logs, use LLM
        logs_without_matches = (
            self.logs
            .filter(lambda t: t.id not in logs_with_matches.id)
            .select(
                log_id=pw.this.log_id,
                content=pw.this.content,
                prompt=self._build_template_prompt(pw.this.content)
            )
            .select(
                pw.this.log_id,
                pw.this.content,
                llm_response=self.model(pw.this.prompt)
            )
        )
        
        # Combine results and evaluate
        all_results = pw.Table.concat(
            logs_with_matches.select(
                log_id=pw.this.log_id,
                content=pw.this.content,
                predicted_template=pw.this.template,
                ground_truth=pw.this.ground_truth,
                inference_time=0.0  # Template matching is fast
            ),
            logs_without_matches.select(
                log_id=pw.this.log_id,
                content=pw.this.content,
                predicted_template=pw.apply(
                    lambda x: json.loads(x)["template"],
                    pw.this.llm_response
                ),
                ground_truth=pw.this.ground_truth,
                inference_time=pw.apply(
                    lambda x: x.get("inference_time", 100.0),
                    pw.this.llm_response
                )
            )
        )
        
        return all_results
    
    def evaluate(self) -> EvaluationMetrics:
        """Evaluate the pipeline results."""
        # Convert results to format expected by metrics
        results_df = pw.debug.compute_and_print(self.results, include_id=False)
        
        predicted_templates = {
            row['log_id']: row['predicted_template']
            for _, row in results_df.iterrows()
        }
        
        inference_times = results_df['inference_time'].tolist()
        
        # Calculate metrics
        metrics = evaluate_parser_output(
            ground_truth_templates=self.dataset.ground_truth_templates,
            predicted_templates=predicted_templates,
            inference_times_ms=inference_times,
            model_name="pathway_pipeline"
        )
        
        # Generate report
        self._generate_report(metrics, predicted_templates)
        
        return metrics
    
    def _generate_report(
        self,
        metrics: EvaluationMetrics,
        predicted_templates: Dict[int, str]
    ) -> None:
        """Generate evaluation report and template files."""
        # Save metrics
        metrics_file = self.output_dir / f"{self.dataset.name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics.dict(), f, indent=2)
        
        # Save templates
        template_file = self.output_dir / f"{self.dataset.name}_templates.csv"
        template_df = pd.DataFrame({
            'log_id': list(predicted_templates.keys()),
            'template': list(predicted_templates.values())
        })
        template_df.to_csv(template_file, index=False)
        
        # Generate markdown report
        report = f"""# Evaluation Report for {self.dataset.name}

## Dataset Statistics
- Total Logs: {metrics.total_logs}
- Unique Templates: {metrics.unique_templates}

## Performance Metrics
- Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}
- Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}
- F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}
- F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}
- Average Inference Time: {metrics.avg_inference_time_ms:.2f}ms
"""
        
        report_file = self.output_dir / f"{self.dataset.name}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)

def main():
    """Main entry point for evaluation."""
    pipeline = EvaluationPipeline(
        base_dir="./data/eval_datasets",
        dataset_type="loghub_2k",
        system="Apache",
        output_dir="./output/eval",
        cache_dir="./cache/eval"
    )
    
    pipeline.setup_pipeline()
    metrics = pipeline.evaluate()
    print(f"Evaluation complete. Results saved to {pipeline.output_dir}")

if __name__ == "__main__":
    main() 