"""Module implementing evaluation metrics for log parsing."""

from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score

@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    grouping_accuracy: float
    parsing_accuracy: float
    f1_grouping_accuracy: float
    f1_template_accuracy: float
    grouping_granularity_distance: float
    parsing_granularity_distance: float
    avg_inference_time_ms: float
    total_logs: int
    unique_templates: int
    model_name: str
    total_api_calls: int = 0
    cache_hit_rate: float = 0.0

def calculate_grouping_accuracy(ground_truth_groups: Dict[str, Set[int]], 
                              predicted_groups: Dict[str, Set[int]]) -> float:
    """Calculate Grouping Accuracy (GA) metric.
    
    GA measures how well the parser groups similar log messages together.
    
    Args:
        ground_truth_groups: Dict mapping template to set of log IDs
        predicted_groups: Dict mapping template to set of log IDs
        
    Returns:
        Grouping accuracy score between 0 and 1
    """
    total_pairs = 0
    correct_pairs = 0
    
    # Convert to list of log IDs for easier processing
    log_ids = list(set.union(*ground_truth_groups.values()))
    
    # Create reverse mappings for efficient lookup
    ground_truth_map = {
        log_id: template 
        for template, ids in ground_truth_groups.items() 
        for log_id in ids
    }
    predicted_map = {
        log_id: template 
        for template, ids in predicted_groups.items() 
        for log_id in ids
    }
    
    # Compare all pairs of log messages
    for i, log_id1 in enumerate(log_ids):
        for log_id2 in log_ids[i+1:]:
            total_pairs += 1
            
            # Check if both pairs are grouped the same way in ground truth and prediction
            same_group_truth = ground_truth_map[log_id1] == ground_truth_map[log_id2]
            same_group_pred = predicted_map[log_id1] == predicted_map[log_id2]
            
            if same_group_truth == same_group_pred:
                correct_pairs += 1
    
    return correct_pairs / total_pairs if total_pairs > 0 else 0.0

def calculate_parsing_accuracy(ground_truth_templates: Dict[int, str],
                             predicted_templates: Dict[int, str]) -> float:
    """Calculate Parsing Accuracy (PA) metric.
    
    PA measures how well the parser extracts the correct templates.
    
    Args:
        ground_truth_templates: Dict mapping log ID to template
        predicted_templates: Dict mapping log ID to template
        
    Returns:
        Parsing accuracy score between 0 and 1
    """
    total_logs = len(ground_truth_templates)
    correct_templates = sum(
        1 for log_id in ground_truth_templates
        if log_id in predicted_templates and 
        ground_truth_templates[log_id] == predicted_templates[log_id]
    )
    
    return correct_templates / total_logs if total_logs > 0 else 0.0

def calculate_f1_scores(ground_truth_groups: Dict[str, Set[int]],
                       predicted_groups: Dict[str, Set[int]]) -> Tuple[float, float]:
    """Calculate F1-scores for Grouping Accuracy (FGA) and Template Accuracy (FTA).
    
    Args:
        ground_truth_groups: Dict mapping template to set of log IDs
        predicted_groups: Dict mapping template to set of log IDs
        
    Returns:
        Tuple of (FGA score, FTA score)
    """
    # Convert groups to labels for sklearn
    log_ids = list(set.union(*ground_truth_groups.values()))
    
    # Create label arrays
    y_true = []
    y_pred = []
    
    # Create reverse mappings
    ground_truth_map = {
        log_id: template 
        for template, ids in ground_truth_groups.items() 
        for log_id in ids
    }
    predicted_map = {
        log_id: template 
        for template, ids in predicted_groups.items() 
        for log_id in ids
    }
    
    # Convert templates to numeric labels
    template_to_label = {}
    current_label = 0
    
    for log_id in log_ids:
        # Ground truth template
        gt_template = ground_truth_map.get(log_id)
        if gt_template not in template_to_label:
            template_to_label[gt_template] = current_label
            current_label += 1
        y_true.append(template_to_label[gt_template])
        
        # Predicted template
        pred_template = predicted_map.get(log_id)
        if pred_template not in template_to_label:
            template_to_label[pred_template] = current_label
            current_label += 1
        y_pred.append(template_to_label[pred_template])
    
    # Calculate F1 scores
    fga = float(f1_score(y_true, y_pred, average='micro'))
    fta = float(f1_score(y_true, y_pred, average='macro'))
    
    return fga, fta

def calculate_granularity_distances(ground_truth_groups: Dict[str, Set[int]],
                                  predicted_groups: Dict[str, Set[int]]) -> Tuple[float, float]:
    """Calculate Grouping and Parsing Granularity Distance (GGD, PGD).
    
    These metrics measure how well the granularity of the parsing matches the ground truth.
    
    Args:
        ground_truth_groups: Dict mapping template to set of log IDs
        predicted_groups: Dict mapping template to set of log IDs
        
    Returns:
        Tuple of (GGD score, PGD score)
    """
    # Calculate group sizes
    gt_sizes = [len(group) for group in ground_truth_groups.values()]
    pred_sizes = [len(group) for group in predicted_groups.values()]
    
    # Calculate mean sizes
    gt_mean = np.mean(gt_sizes)
    pred_mean = np.mean(pred_sizes)
    
    # Calculate standard deviations
    gt_std = np.std(gt_sizes)
    pred_std = np.std(pred_sizes)
    
    # Calculate distances
    ggd = float(abs(gt_mean - pred_mean) / float(max(float(gt_mean), float(pred_mean))))
    pgd = float(abs(gt_std - pred_std) / float(max(float(gt_std), float(pred_std))))
    
    return ggd, pgd

def evaluate_parser_output(ground_truth_templates: Dict[int, str],
                         predicted_templates: Dict[int, str],
                         inference_times_ms: List[float],
                         model_name: str) -> EvaluationMetrics:
    """Evaluate parser output using all metrics.
    
    Args:
        ground_truth_templates: Dict mapping log ID to template
        predicted_templates: Dict mapping log ID to template
        inference_times_ms: List of inference times in milliseconds
        model_name: Name of the model used for parsing
        
    Returns:
        EvaluationMetrics object containing all calculated metrics
    """
    # Convert templates to groups
    ground_truth_groups = defaultdict(set)
    predicted_groups = defaultdict(set)
    
    for log_id, template in ground_truth_templates.items():
        ground_truth_groups[template].add(log_id)
    
    for log_id, template in predicted_templates.items():
        predicted_groups[template].add(log_id)
    
    # Calculate all metrics
    ga = calculate_grouping_accuracy(ground_truth_groups, predicted_groups)
    pa = calculate_parsing_accuracy(ground_truth_templates, predicted_templates)
    fga, fta = calculate_f1_scores(ground_truth_groups, predicted_groups)
    ggd, pgd = calculate_granularity_distances(ground_truth_groups, predicted_groups)
    
    return EvaluationMetrics(
        grouping_accuracy=float(ga),
        parsing_accuracy=float(pa),
        f1_grouping_accuracy=float(fga),
        f1_template_accuracy=float(fta),
        grouping_granularity_distance=float(ggd),
        parsing_granularity_distance=float(pgd),
        avg_inference_time_ms=float(np.mean(inference_times_ms)),
        total_logs=len(ground_truth_templates),
        unique_templates=len(set(ground_truth_templates.values())),
        model_name=model_name
    ) 