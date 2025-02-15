"""
Evaluation pipeline implementation using Pathway.
"""

import json
import logging
from pathlib import Path
from typing import Dict
from datetime import datetime, UTC
import pandas as pd
from dataclasses import asdict
import os

import pathway as pw

from src.eval.datasets import DatasetLoader, LogDataset
from src.eval.metrics import evaluate_parser_output, EvaluationMetrics
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer

# Set up logging
logger = logging.getLogger(__name__)

@pw.udf
def parse_timestamp(timestamp_str: str) -> pw.DateTimeUtc:
    """Parse timestamp string to Pathway UTC datetime."""
    try:
        logger.debug(f"Parsing timestamp: {timestamp_str}")
        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return pw.DateTimeUtc(dt)
    except (ValueError, AttributeError) as e:
        logger.error(f"Error parsing timestamp {timestamp_str}: {e}")
        return pw.DateTimeUtc(datetime.now(UTC))

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
        logger.info("Initializing EvaluationPipeline")
        logger.info(f"Base directory: {base_dir}")
        logger.info(f"Dataset type: {dataset_type}")
        logger.info(f"System: {system}")
        logger.info(f"LLM model: {llm_model}")
        logger.info(f"LLM API base: {llm_api_base}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Batch size: {batch_size}")
        
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.system = system
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Initialize Ollama analyzer
        logger.info("Initializing Ollama analyzer")
        self.ollama_analyzer = create_ollama_analyzer(
            base_url=llm_api_base,
            model_id=llm_model.split('/')[-1],
            config={"temperature": 0.1}
        )
        
        # Initialize log parser
        logger.info("Initializing LogParserLLM")
        self.log_parser = LogParserLLM(
            ollama_client=self.ollama_analyzer,
            similarity_threshold=similarity_threshold
        )
        
        # Load dataset
        logger.info("Loading dataset")
        self.dataset = self._load_dataset()
        
        # Create output directories
        logger.info("Creating output directories")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_dataset(self) -> LogDataset:
        """Load dataset for evaluation."""
        logger.info(f"Loading dataset from {self.base_dir}")
        loader = DatasetLoader(self.base_dir)
        dataset = loader.load_dataset(
            system=self.system,
            dataset_type=self.dataset_type
        )
        logger.info(f"Loaded dataset {dataset.name} with {dataset.size} logs")
        return dataset
    
    def setup_pipeline(self) -> None:
        """Set up the Pathway evaluation pipeline."""
        logger.info("Setting up evaluation pipeline")
        
        # Convert dataset to Pathway table
        logger.info("Converting dataset to Pathway table")
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
        logger.info(f"Created Pathway table with {len(df)} rows")
        
        # Add parsed timestamp
        logger.info("Processing timestamps")
        self.logs = self.logs.select(
            log_id=pw.this.log_id,
            content=pw.this.content,
            timestamp=parse_timestamp(pw.this.timestamp),
            ground_truth=pw.this.ground_truth
        )
        
        # Process logs through LogParserLLM
        logger.info("Processing logs")
        self.results = self._process_logs()
        
    def _process_logs(self) -> pw.Table:
        """Process logs using LogParserLLM."""
        # Process logs one at a time
        processed_logs = []
        total_logs = len(self.dataset.raw_logs)
        logger.info(f"Processing {total_logs} logs")
        
        for i, log in enumerate(self.dataset.raw_logs):
            if i % 100 == 0:  # Log progress every 100 logs
                logger.info(f"Processing log {i+1}/{total_logs} ({((i+1)/total_logs)*100:.1f}%)")
            result = self.log_parser.parse_log(log, i)
            processed_logs.append(result)
        
        # Convert results to DataFrame
        logger.info("Converting results to DataFrame")
        results_df = pd.DataFrame({
            'log_id': range(len(processed_logs)),
            'content': self.dataset.raw_logs,
            'predicted_template': [r[0] for r in processed_logs],  # template is first element
            'ground_truth': [
                self.dataset.ground_truth_templates[i]
                for i in range(len(processed_logs))
            ],
            'inference_time': [0 for _ in processed_logs]  # No inference time in parse_log
        })
        
        logger.info(f"Processed {len(processed_logs)} logs successfully")
        return pw.debug.table_from_pandas(results_df)
    
    def evaluate(self) -> EvaluationMetrics:
        """Evaluate the pipeline results."""
        logger.info("Starting evaluation")
        
        # Convert results to format expected by metrics
        logger.info("Converting results for metrics calculation")
        temp_csv = os.path.join(self.cache_dir, "temp_results.csv")
        pw.io.csv.write(self.results, temp_csv)
        results_df = pd.read_csv(temp_csv)
        os.remove(temp_csv)  # Clean up
        
        predicted_templates = {
            row['log_id']: row['predicted_template']
            for _, row in results_df.iterrows()
        }
        
        inference_times = results_df['inference_time'].tolist()
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics")
        metrics = evaluate_parser_output(
            ground_truth_templates=self.dataset.ground_truth_templates,
            predicted_templates=predicted_templates,
            inference_times_ms=inference_times,
            model_name="pathway_pipeline"
        )
        
        # Generate report
        logger.info("Generating evaluation report")
        self._generate_report(metrics, predicted_templates)
        
        # Log metrics summary
        logger.info("Evaluation Results:")
        logger.info(f"  Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}")
        logger.info(f"  Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}")
        logger.info(f"  F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}")
        logger.info(f"  F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}")
        logger.info(f"  Average Inference Time: {metrics.avg_inference_time_ms:.2f}ms")
        
        return metrics
    
    def _generate_report(
        self,
        metrics: EvaluationMetrics,
        predicted_templates: Dict[int, str]
    ) -> None:
        """Generate evaluation report and template files."""
        # Save metrics
        metrics_file = self.output_dir / f"{self.dataset.name}_metrics.json"
        logger.info(f"Saving metrics to {metrics_file}")
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Save templates
        template_file = self.output_dir / f"{self.dataset.name}_templates.csv"
        logger.info(f"Saving templates to {template_file}")
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
        logger.info(f"Saving report to {report_file}")
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