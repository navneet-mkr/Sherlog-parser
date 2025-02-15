"""
Evaluation pipeline implementation using Pathway.
"""

import json
import time
from pathlib import Path
from typing import Dict
from datetime import datetime
import pandas as pd

import pathway as pw

from src.eval.datasets import DatasetLoader, LogDataset
from src.eval.metrics import evaluate_parser_output, EvaluationMetrics
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer

@pw.udf
def parse_timestamp(timestamp_str: str) -> pw.DateTimeUtc:
    """Parse timestamp string to Pathway UTC datetime."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return pw.DateTimeUtc.from_python(dt)
    except (ValueError, AttributeError):
        # Return a default timestamp if parsing fails
        return pw.DateTimeUtc.from_python(datetime.utcnow())

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
    
    def setup_pipeline(self) -> None:
        """Set up the Pathway evaluation pipeline."""
        # Convert dataset to Pathway table
        df = pd.DataFrame({
            'log_id': range(len(self.dataset.raw_logs)),
            'content': self.dataset.raw_logs,
            'timestamp': [datetime.utcnow().isoformat() for _ in self.dataset.raw_logs],  # Add timestamps
            'ground_truth': [
                self.dataset.ground_truth_templates[i]
                for i in range(len(self.dataset.raw_logs))
            ]
        })
        self.logs = pw.debug.table_from_pandas(df)
        
        # Add parsed timestamp
        self.logs = self.logs.select(
            log_id=pw.this.log_id,
            content=pw.this.content,
            timestamp=parse_timestamp(pw.this.timestamp),
            ground_truth=pw.this.ground_truth
        )
        
        # Process logs through LogParserLLM
        self.results = self._process_logs()
        
    def _process_logs(self) -> pw.Table:
        """Process logs using LogParserLLM."""
        # Process logs in batches
        processed_logs = []
        for i in range(0, len(self.dataset.raw_logs), self.batch_size):
            batch = self.dataset.raw_logs[i:i + self.batch_size]
            results = self.log_parser.process_logs(batch)
            processed_logs.extend(results)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame({
            'log_id': range(len(processed_logs)),
            'content': self.dataset.raw_logs,
            'predicted_template': [r.template for r in processed_logs],
            'ground_truth': [
                self.dataset.ground_truth_templates[i]
                for i in range(len(processed_logs))
            ],
            'inference_time': [r.inference_time_ms for r in processed_logs]
        })
        
        return pw.debug.table_from_pandas(results_df)
    
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