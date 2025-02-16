"""
Simplified evaluation implementation without Pathway dependency.
"""

import json
import logging
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import re

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
        batch_size: int = 32,
        track_api_calls: bool = False
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
            track_api_calls: Whether to track API calls
        """
        logger.info("Initializing Evaluator")
        logger.info(f"Dataset: {system}/{dataset_type}")
        logger.info(f"Model: {llm_model}")
        
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.track_api_calls = track_api_calls
        self.api_calls = 0
        self.cache_hits = 0
        
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
            batch_size=batch_size,
            track_api_calls=track_api_calls
        )
    
    def _normalize_template(self, template: str) -> str:
        """Normalize template by converting all variable placeholders to <*>.
        
        Args:
            template: Template string with variable placeholders
            
        Returns:
            Normalized template with standardized placeholders
        """
        # Convert all variable placeholders to <*>
        normalized = re.sub(r'<[^>]+>', '<*>', template)
        return normalized

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
        
        # Normalize templates before comparison
        predicted_templates = {
            row['log_id']: self._normalize_template(row['template'])
            for _, row in parsed_logs_df.iterrows()
        }
        
        ground_truth_normalized = {
            log_id: self._normalize_template(template)
            for log_id, template in self.dataset.ground_truth_templates.items()
        }
        
        # Calculate metrics
        logger.info("Calculating metrics")
        metrics = evaluate_parser_output(
            ground_truth_templates=ground_truth_normalized,
            predicted_templates=predicted_templates,
            inference_times_ms=[0] * len(predicted_templates),
            model_name=self.parser.model_name
        )
        
        # Add API call metrics if tracking is enabled
        if self.track_api_calls:
            metrics_dict = asdict(metrics)
            metrics_dict["total_api_calls"] = self.parser.get_api_calls()
            metrics_dict["cache_hit_rate"] = self.parser.get_cache_hit_rate()
            metrics = EvaluationMetrics(**metrics_dict)
        
        # Generate report
        self._generate_report(metrics, templates_df)
        
        # Log metrics
        logger.info("Evaluation Results:")
        logger.info(f"  Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}")
        logger.info(f"  Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}")
        logger.info(f"  F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}")
        logger.info(f"  F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}")
        if self.track_api_calls:
            logger.info(f"  Total API Calls: {metrics.total_api_calls}")
            logger.info(f"  Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
        
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