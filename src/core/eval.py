"""
Simplified evaluation implementation without Pathway dependency.
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from src.core.parser_service import ParserService
from src.eval.datasets import DatasetLoader
from src.eval.metrics import evaluate_parser_output, EvaluationMetrics

logger = logging.getLogger(__name__)

class Evaluator:
    """Log parser evaluator."""
    
    def __init__(
        self,
        base_dir: str,
        dataset_type: str,
        system: str,
        llm_model: str = "ollama/mistral",
        llm_api_base: str = "http://localhost:11434",
        output_dir: str = "./output/eval",
        cache_dir: str = "./cache/eval",
        similarity_threshold: float = 0.8,
        batch_size: int = 32
    ):
        """Initialize evaluator.
        
        Args:
            base_dir: Base directory containing datasets
            dataset_type: Type of dataset to evaluate
            system: System to evaluate
            llm_model: Name of the LLM model to use
            llm_api_base: Base URL for LLM API
            output_dir: Directory to save results
            cache_dir: Directory for caching
            similarity_threshold: Threshold for template matching
            batch_size: Number of logs to process in each batch
        """
        logger.info("Initializing Evaluator")
        logger.info(f"Dataset: {system}/{dataset_type}")
        logger.info(f"Model: {llm_model}")
        
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        logger.info("Loading dataset")
        loader = DatasetLoader(base_dir)
        self.dataset = loader.load_dataset(
            system=system,
            dataset_type=dataset_type
        )
        logger.info(f"Loaded {self.dataset.size} logs")
        
        # Initialize parser
        self.parser = ParserService(
            llm_model=llm_model,
            llm_api_base=llm_api_base,
            similarity_threshold=similarity_threshold,
            batch_size=batch_size
        )
    
    def evaluate(self) -> EvaluationMetrics:
        """Run evaluation.
        
        Returns:
            Evaluation metrics
        """
        logger.info("Starting evaluation")
        
        # Create temporary log file
        temp_file = self.cache_dir / "temp_logs.csv"
        pd.DataFrame({
            'content': self.dataset.raw_logs
        }).to_csv(temp_file, index=False)
        
        # Parse logs
        parsed_logs_df, templates_df = self.parser.parse_logs(
            log_file=str(temp_file),
            output_dir=str(self.output_dir)
        )
        
        # Convert results for metrics
        predicted_templates = {
            row['log_id']: row['template'] 
            for _, row in parsed_logs_df.iterrows()
        }
        
        # Calculate metrics
        logger.info("Calculating metrics")
        metrics = evaluate_parser_output(
            ground_truth_templates=self.dataset.ground_truth_templates,
            predicted_templates=predicted_templates,
            inference_times_ms=[0] * len(predicted_templates),  # Not measuring time in simplified version
            model_name="simplified_parser"
        )
        
        # Generate report
        self._generate_report(metrics, templates_df)
        
        # Log metrics
        logger.info("Evaluation Results:")
        logger.info(f"  Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}")
        logger.info(f"  Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}")
        logger.info(f"  F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}")
        logger.info(f"  F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}")
        
        return metrics
    
    def _generate_report(
        self,
        metrics: EvaluationMetrics,
        templates_df: pd.DataFrame
    ) -> None:
        """Generate evaluation report.
        
        Args:
            metrics: Evaluation metrics
            templates_df: DataFrame with templates
        """
        # Save metrics
        metrics_file = self.output_dir / f"{self.dataset.name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info(f"Saved metrics to {metrics_file}")
        
        # Save templates
        template_file = self.output_dir / f"{self.dataset.name}_templates.csv"
        templates_df.to_csv(template_file, index=False)
        logger.info(f"Saved templates to {template_file}")
        
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

## Template Analysis
Top 10 Templates by Frequency:
{templates_df.nlargest(10, 'count')[['template_id', 'template', 'count']].to_string()}
"""
        
        report_file = self.output_dir / f"{self.dataset.name}_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {report_file}")

def main():
    """Main entry point for evaluation."""
    evaluator = Evaluator(
        base_dir="./data/eval_datasets",
        dataset_type="loghub_2k",
        system="Apache"
    )
    metrics = evaluator.evaluate()
    print(f"Evaluation complete. Results saved to {evaluator.output_dir}")

if __name__ == "__main__":
    main() 