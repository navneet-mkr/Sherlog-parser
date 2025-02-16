"""Module implementing evaluation metrics for log parsing."""

from typing import Dict, List, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score
import pandas as pd
import itertools
from Levenshtein import distance

@dataclass
class EvaluationMetrics:
    """Evaluation metrics for log parsing."""
    system: str
    dataset: str
    total_logs: int
    unique_templates: int
    ground_truth_templates: int
    grouping_accuracy: float = 0.0
    parsing_accuracy: float = 0.0
    f1_grouping_accuracy: float = 0.0
    f1_template_accuracy: float = 0.0
    grouping_granularity_distance: float = 0.0
    parsing_granularity_distance: float = 0.0
    avg_inference_time_ms: float = 0.0
    model_name: str = ""
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

def _compare_templates(template1: str, template2: str) -> bool:
    """Compare two templates for structural equality.
    
    Args:
        template1: First template string
        template2: Second template string
        
    Returns:
        True if templates match structurally
    """
    # Split templates into parts
    parts1 = template1.split('<*>')
    parts2 = template2.split('<*>')
    
    # Check if they have same number of variable placeholders
    if len(parts1) != len(parts2):
        return False
        
    # Compare constant parts
    for p1, p2 in zip(parts1, parts2):
        if p1.strip() != p2.strip():
            return False
            
    return True

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
        _compare_templates(ground_truth_templates[log_id], predicted_templates[log_id])
    )
    
    return correct_templates / total_logs if total_logs > 0 else 0.0

def calculate_f1_score(predicted: Dict[int, str], ground_truth: Dict[int, str]) -> float:
    """Calculate F1 score for grouping."""
    # Create binary matrix for pairs
    n_logs = len(predicted)
    true_pairs = np.zeros((n_logs, n_logs))
    pred_pairs = np.zeros((n_logs, n_logs))
    
    # Fill matrices
    for i, j in itertools.combinations(range(n_logs), 2):
        true_pairs[i,j] = true_pairs[j,i] = ground_truth[i] == ground_truth[j]
        pred_pairs[i,j] = pred_pairs[j,i] = predicted[i] == predicted[j]
    
    # Flatten and calculate F1
    true_flat = true_pairs.flatten()
    pred_flat = pred_pairs.flatten()
    
    return float(f1_score(true_flat, pred_flat))

def calculate_template_f1_score(predicted: Dict[int, str], ground_truth: Dict[int, str]) -> float:
    """Calculate F1 score for template matching."""
    # Get unique templates
    true_templates = set(ground_truth.values())
    pred_templates = set(predicted.values())
    
    # Calculate precision and recall
    true_positives = len(true_templates & pred_templates)
    precision = true_positives / len(pred_templates) if pred_templates else 0
    recall = true_positives / len(true_templates) if true_templates else 0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

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

def calculate_template_similarity(template1: str, template2: str) -> float:
    """Calculate similarity between two templates using normalized Levenshtein distance.
    
    Args:
        template1: First template string
        template2: Second template string
        
    Returns:
        Similarity score between 0 and 1
    """
    max_len = max(len(template1), len(template2))
    if max_len == 0:
        return 1.0
    return 1 - (distance(template1, template2) / max_len)

def evaluate_parser_output(results_df: pd.DataFrame,
                        ground_truth_df: pd.DataFrame,
                        system: str,
                        dataset_type: str = "loghub_2k") -> Dict[str, Union[str, int, float]]:
    """Evaluate parser output against ground truth.
    
    Args:
        results_df: DataFrame with parser results
        ground_truth_df: DataFrame with ground truth
        system: System name
        dataset_type: Dataset type
        
    Returns:
        Dictionary of evaluation metrics
    """
    total_logs = len(results_df)
    
    # Extract templates
    predicted_templates = results_df['ParsedTemplate'].tolist()
    ground_truth_templates = ground_truth_df['EventTemplate'].tolist()
    
    # Get unique templates
    unique_predicted = len(set(predicted_templates))
    unique_ground_truth = len(set(ground_truth_templates))
    
    # Calculate grouping accuracy
    correct_groups = sum(1 for i, j in itertools.combinations(range(total_logs), 2)
        if (results_df['ParsedTemplate'].iloc[i] == results_df['ParsedTemplate'].iloc[j]) ==
           (ground_truth_df['EventTemplate'].iloc[i] == ground_truth_df['EventTemplate'].iloc[j]))
    total_pairs = total_logs * (total_logs - 1) // 2
    grouping_accuracy = correct_groups / total_pairs if total_pairs > 0 else 0
    
    # Calculate edit distance based similarity
    template_similarities = []
    for pred, truth in zip(predicted_templates, ground_truth_templates):
        similarity = calculate_template_similarity(pred, truth)
        template_similarities.append(similarity)
    
    avg_similarity = sum(template_similarities) / len(template_similarities)
    
    return {
        "system": system,
        "dataset": dataset_type,
        "total_logs": total_logs,
        "unique_templates_predicted": unique_predicted,
        "unique_templates_ground_truth": unique_ground_truth,
        "grouping_accuracy": grouping_accuracy,
        "template_similarity": avg_similarity
    } 