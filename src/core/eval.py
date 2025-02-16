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
        
        self.base_dir = Path(base_dir)
        self.dataset_type = dataset_type
        self.system = system
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.track_api_calls = track_api_calls
        self.api_calls = 0
        self.cache_hits = 0
        
        # Load dataset
        logger.info("Loading dataset")
        loader = DatasetLoader(str(self.base_dir))
        self.dataset = loader.load_dataset(
            system=system,
            dataset_type=dataset_type
        )
        logger.info(f"Loaded {self.dataset.size} logs")
        
        # Initialize parser
        self.parser = ParserService(
            llm_model=self.llm_model,
            llm_api_base=self.llm_api_base,
            similarity_threshold=self.similarity_threshold,
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
        """Run evaluation on the dataset."""
        logger.info("Starting evaluation")
        
        # Load dataset
        dataset = DatasetLoader(str(self.base_dir))
        logs_df = dataset.load_logs(self.system, self.dataset_type)
        ground_truth_df = dataset.load_templates(self.system, self.dataset_type)
        
        # Initialize parser service
        parser = ParserService(
            llm_model=self.llm_model,
            llm_api_base=self.llm_api_base,
            similarity_threshold=self.similarity_threshold
        )
        
        # Process logs in batches
        parsed_templates = []
        parsed_parameters = []
        total_logs = len(logs_df)
        
        logger.info(f"Processing {total_logs} logs in batches of {self.batch_size}")
        
        # Prepare log batches with IDs
        log_batches = [
            list(zip(
                logs_df['Content'].iloc[i:i + self.batch_size],
                range(i, min(i + self.batch_size, total_logs))
            ))
            for i in range(0, total_logs, self.batch_size)
        ]
        
        # Process each batch
        for batch_idx, batch in enumerate(log_batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(log_batches)}")
            
            try:
                # Process batch in parallel
                results = parser.parse_logs_batch(batch, self.batch_size)
                
                # Collect results
                batch_templates, batch_parameters = zip(*results)
                parsed_templates.extend(batch_templates)
                parsed_parameters.extend(batch_parameters)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {str(e)}")
                # Add empty results for failed batch
                parsed_templates.extend([''] * len(batch))
                parsed_parameters.extend([{}] * len(batch))
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Content': logs_df['Content'],
            'ParsedTemplate': parsed_templates,
            'Parameters': parsed_parameters
        })
        
        # Normalize templates for comparison
        results_df['ParsedTemplate'] = results_df['ParsedTemplate'].apply(self._normalize_template)
        ground_truth_df['EventTemplate'] = ground_truth_df['EventTemplate'].apply(self._normalize_template)
        
        # Calculate metrics
        metrics_dict = evaluate_parser_output(
            results_df,
            ground_truth_df,
            self.system,
            self.dataset_type
        )
        
        # Convert values to proper types
        typed_metrics = {
            'system': str(metrics_dict['system']),
            'dataset': str(metrics_dict['dataset']),
            'total_logs': int(metrics_dict['total_logs']),
            'unique_templates': int(metrics_dict['unique_templates']),
            'ground_truth_templates': int(metrics_dict['ground_truth_templates']),
            'grouping_accuracy': float(metrics_dict['grouping_accuracy']),
            'parsing_accuracy': float(metrics_dict['parsing_accuracy']),
            'f1_grouping_accuracy': float(metrics_dict['f1_grouping_accuracy']),
            'f1_template_accuracy': float(metrics_dict['f1_template_accuracy']),
            'grouping_granularity_distance': float(metrics_dict['grouping_granularity_distance'])
        }
        
        # Convert to EvaluationMetrics object
        metrics = EvaluationMetrics(**typed_metrics)
        
        # Save results
        self._save_results(results_df, metrics)
        
        return metrics
    
    def _save_results(self, results_df: pd.DataFrame, metrics: EvaluationMetrics) -> None:
        """Save evaluation results to files."""
        # Save parsed templates
        results_df.to_csv(
            self.output_dir / f"{self.system}_{self.dataset_type}_templates.csv",
            index=False
        )
        
        # Save metrics
        with open(self.output_dir / f"{self.system}_{self.dataset_type}_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
            
        # Generate report
        report_path = self.output_dir / f"{self.system}_{self.dataset_type}_report.md"
        with open(report_path, 'w') as f:
            f.write(f"# Evaluation Report for {self.system} {self.dataset_type}\n\n")
            f.write(f"## Metrics\n")
            f.write(f"- Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}\n")
            f.write(f"- Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}\n")
            f.write(f"- F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}\n")
            f.write(f"- F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}\n")
            if self.track_api_calls:
                f.write(f"- Total API Calls: {metrics.total_api_calls}\n")
                f.write(f"- Cache Hit Rate: {metrics.cache_hit_rate:.1%}\n")

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