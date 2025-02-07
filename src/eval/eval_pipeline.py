"""Evaluation pipeline for LogParser-LLM."""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd

from dagster import (
    job, op, In, Out, Nothing,
    AssetMaterialization, MetadataValue,
    DagsterType, String, Int, Bool, Float,
    ResourceDefinition
)

from src.eval.datasets import DatasetLoader, LogDataset, DEFAULT_TEST_DATASETS
from src.eval.metrics import evaluate_parser_output, EvaluationMetrics
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer, VariableType

# Custom Dagster types
def is_valid_dataset(_, value: LogDataset) -> bool:
    """Validate dataset object."""
    return isinstance(value, LogDataset)

LogDatasetType = DagsterType(
    name="LogDataset",
    type_check_fn=is_valid_dataset,
    description="A log dataset with ground truth"
)

def is_valid_metrics(_, value: EvaluationMetrics) -> bool:
    """Validate metrics object."""
    return isinstance(value, EvaluationMetrics)

MetricsType = DagsterType(
    name="EvaluationMetrics",
    type_check_fn=is_valid_metrics,
    description="Evaluation metrics for a dataset"
)

# Pipeline operations
@op(
    config_schema={
        "base_dir": String,
        "dataset_type": String,
        "system": String
    },
    out=Out(LogDatasetType)
)
def load_dataset(context) -> LogDataset:
    """Load a dataset for evaluation."""
    try:
        loader = DatasetLoader(context.op_config["base_dir"])
        dataset = loader.load_dataset(
            system=context.op_config["system"],
            dataset_type=context.op_config["dataset_type"]
        )
        
        context.log.info(
            f"Loaded dataset {dataset.name} with {dataset.size} logs "
            f"and {len(set(dataset.ground_truth_templates.values()))} unique templates"
        )
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key=f"dataset_{dataset.name}",
                description=f"Loaded dataset {dataset.name}",
                metadata={
                    "total_logs": MetadataValue.int(dataset.size),
                    "unique_templates": MetadataValue.int(
                        len(set(dataset.ground_truth_templates.values()))
                    )
                }
            )
        )
        
        return dataset
    except Exception as e:
        context.log.error(f"Failed to load dataset: {str(e)}")
        raise

@op(
    ins={"dataset": In(LogDatasetType)},
    config_schema={
        "ollama_base_url": String,
        "model_name": String,
        "similarity_threshold": Float,
        "batch_size": Int,
        "cache_dir": String
    },
    out={"templates": Out(dict), "inference_times": Out(list)}
)
def parse_dataset(context, dataset: LogDataset) -> Tuple[Dict[int, str], List[float]]:
    """Parse logs in dataset using LogParser-LLM."""
    try:
        # Check cache first
        cache_dir = Path(context.op_config["cache_dir"])
        cache_file = cache_dir / f"{dataset.name}_{context.op_config['model_name']}_results.json"
        
        if cache_file.exists():
            context.log.info(f"Loading cached results from {cache_file}")
            with open(cache_file) as f:
                cached = json.load(f)
                return cached["templates"], cached["inference_times"]
        
        # Create Ollama client
        ollama_client = create_ollama_analyzer(
            base_url=context.op_config["ollama_base_url"],
            model_id=context.op_config["model_name"],
            config={}
        )
        
        # Initialize LogParser-LLM
        parser = LogParserLLM(
            ollama_client=ollama_client,
            similarity_threshold=context.op_config["similarity_threshold"]
        )
        
        # Process logs in batches
        batch_size = context.op_config["batch_size"]
        templates = {}
        inference_times = []
        
        for i in range(0, len(dataset.raw_logs), batch_size):
            batch = dataset.raw_logs[i:i + batch_size]
            for log_id, log_message in enumerate(batch, start=i):
                start_time = time.time()
                template, _ = parser.parse_log(log_message, log_id)
                inference_time = (time.time() - start_time) * 1000  # Convert to ms
                
                templates[log_id] = template
                inference_times.append(inference_time)
                
            context.log.info(f"Processed batch {i//batch_size + 1}")
        
        # Cache results
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump({
                "templates": templates,
                "inference_times": inference_times
            }, f)
        
        return templates, inference_times
    except Exception as e:
        context.log.error(f"Failed to parse dataset: {str(e)}")
        raise

@op(
    ins={
        "dataset": In(LogDatasetType),
        "templates": In(dict),
        "inference_times": In(list)
    },
    config_schema={
        "model_name": String
    },
    out=Out(MetricsType)
)
def evaluate_results(context, dataset: LogDataset, 
                    templates: Dict[int, str],
                    inference_times: List[float]) -> EvaluationMetrics:
    """Evaluate parsing results using all metrics."""
    try:
        metrics = evaluate_parser_output(
            ground_truth_templates=dataset.ground_truth_templates,
            predicted_templates=templates,
            inference_times_ms=inference_times,
            model_name=context.op_config["model_name"]
        )
        
        # Log metrics
        context.log.info(f"Evaluation results for {dataset.name}:")
        context.log.info(f"  Grouping Accuracy (GA): {metrics.grouping_accuracy:.4f}")
        context.log.info(f"  Parsing Accuracy (PA): {metrics.parsing_accuracy:.4f}")
        context.log.info(f"  F1 Grouping Accuracy (FGA): {metrics.f1_grouping_accuracy:.4f}")
        context.log.info(f"  F1 Template Accuracy (FTA): {metrics.f1_template_accuracy:.4f}")
        context.log.info(f"  Grouping Granularity Distance (GGD): {metrics.grouping_granularity_distance:.4f}")
        context.log.info(f"  Parsing Granularity Distance (PGD): {metrics.parsing_granularity_distance:.4f}")
        context.log.info(f"  Average Inference Time: {metrics.avg_inference_time_ms:.2f}ms")
        
        # Record asset materialization with markdown table
        markdown_table = f"""
        | Metric | Value |
        |--------|-------|
        | Dataset | {dataset.name} |
        | Model | {metrics.model_name} |
        | Total Logs | {metrics.total_logs} |
        | Unique Templates | {metrics.unique_templates} |
        | Grouping Accuracy (GA) | {metrics.grouping_accuracy:.4f} |
        | Parsing Accuracy (PA) | {metrics.parsing_accuracy:.4f} |
        | F1 Grouping Accuracy (FGA) | {metrics.f1_grouping_accuracy:.4f} |
        | F1 Template Accuracy (FTA) | {metrics.f1_template_accuracy:.4f} |
        | Grouping Granularity Distance (GGD) | {metrics.grouping_granularity_distance:.4f} |
        | Parsing Granularity Distance (PGD) | {metrics.parsing_granularity_distance:.4f} |
        | Average Inference Time (ms) | {metrics.avg_inference_time_ms:.2f} |
        """
        
        context.log_event(
            AssetMaterialization(
                asset_key=f"metrics_{dataset.name}_{metrics.model_name}",
                description=f"Evaluation metrics for {dataset.name}",
                metadata={
                    "grouping_accuracy": MetadataValue.float(metrics.grouping_accuracy),
                    "parsing_accuracy": MetadataValue.float(metrics.parsing_accuracy),
                    "f1_grouping_accuracy": MetadataValue.float(metrics.f1_grouping_accuracy),
                    "f1_template_accuracy": MetadataValue.float(metrics.f1_template_accuracy),
                    "grouping_granularity_distance": MetadataValue.float(metrics.grouping_granularity_distance),
                    "parsing_granularity_distance": MetadataValue.float(metrics.parsing_granularity_distance),
                    "avg_inference_time_ms": MetadataValue.float(metrics.avg_inference_time_ms),
                    "total_logs": MetadataValue.int(metrics.total_logs),
                    "unique_templates": MetadataValue.int(metrics.unique_templates),
                    "model_name": MetadataValue.text(metrics.model_name),
                    "report": MetadataValue.md(markdown_table)
                }
            )
        )
        
        return metrics
    except Exception as e:
        context.log.error(f"Failed to evaluate results: {str(e)}")
        raise

@op(
    ins={
        "dataset": In(LogDatasetType),
        "templates": In(dict),
        "inference_times": In(list)
    },
    config_schema={
        "output_dir": String,
        "model_name": String
    },
    out=Out(Nothing)
)
def generate_template_file(context, dataset: LogDataset,
                         templates: Dict[int, str],
                         inference_times: List[float]) -> None:
    """Generate a template file for debugging purposes."""
    try:
        output_dir = Path(context.op_config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create template file
        template_file = output_dir / f"{dataset.name}_{context.op_config['model_name']}_templates.csv"
        
        # Convert typed variables to simple <*> format
        var_pattern = r'<(' + '|'.join(vtype.value for vtype in VariableType) + r')>'
        simplified_templates = {
            log_id: re.sub(var_pattern, '<*>', template)
            for log_id, template in templates.items()
        }
        
        # Group logs by template
        template_groups = {}
        for log_id, template in simplified_templates.items():
            if template not in template_groups:
                template_groups[template] = []
            template_groups[template].append(log_id)
        
        # Write in the same format as ground truth template file
        with open(template_file, 'w') as f:
            f.write("EventId,EventTemplate\n")
            for i, (template, log_ids) in enumerate(template_groups.items(), 1):
                event_id = f"E{i}"
                f.write(f"{event_id},{template}\n")
        
        # Also generate a detailed debug file
        debug_file = output_dir / f"{dataset.name}_{context.op_config['model_name']}_debug.txt"
        with open(debug_file, 'w') as f:
            f.write(f"# Generated templates for {dataset.name} using {context.op_config['model_name']}\n")
            f.write(f"# Total logs: {len(dataset.raw_logs)}\n")
            f.write(f"# Unique templates: {len(template_groups)}\n\n")
            
            for template, log_ids in template_groups.items():
                f.write(f"Template: {template}\n")
                f.write(f"Count: {len(log_ids)}\n")
                f.write("Example logs:\n")
                for log_id in log_ids[:5]:
                    f.write(f"  {dataset.raw_logs[log_id]}\n")
                f.write("\n")
        
        context.log.info(f"Generated template files at {template_file} and {debug_file}")
        
        # Record asset materialization
        context.log_event(
            AssetMaterialization(
                asset_key=f"template_file_{dataset.name}_{context.op_config['model_name']}",
                description=f"Generated template files for {dataset.name}",
                metadata={
                    "template_file": MetadataValue.path(str(template_file)),
                    "debug_file": MetadataValue.path(str(debug_file)),
                    "total_templates": MetadataValue.int(len(template_groups)),
                    "total_logs": MetadataValue.int(len(dataset.raw_logs))
                }
            )
        )
    except Exception as e:
        context.log.error(f"Failed to generate template file: {str(e)}")
        raise

@job
def evaluate_logparser_llm():
    """Main evaluation job for LogParser-LLM."""
    # Load and evaluate single dataset
    dataset = load_dataset()
    templates, inference_times = parse_dataset(dataset)
    evaluate_results(dataset, templates, inference_times)
    generate_template_file(dataset, templates, inference_times) 