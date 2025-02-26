#!/usr/bin/env python3
# Python benchmark script for comparing with C++ implementation

import sys
import time
import random
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set

# Mock implementations of the Python classes for benchmarking
# These are simplified versions just for comparison purposes

class TemplateMatcher:
    """Python implementation of template matcher for benchmarking."""
    
    def __init__(self, similarity_threshold: float = 0.8, max_examples: int = 5):
        self.similarity_threshold = similarity_threshold
        self.max_examples = max_examples
    
    def _tokenize(self, template_str: str) -> List[str]:
        """Tokenize template into words."""
        return template_str.split()
    
    def _get_variable_positions(self, tokens: List[str]) -> Set[int]:
        """Get positions of variable tokens in template."""
        variable_types = ['OID', 'LOI', 'OBN', 'TID', 'SID', 'TDA', 'CRS', 'OBA', 'STC', 'OTHER_PARAMS']
        return {i for i, token in enumerate(tokens) 
                if any(f"<{vtype}>" in token for vtype in variable_types)}
    
    def _calculate_token_similarity(
        self,
        tokens1: List[str],
        tokens2: List[str],
        var_positions1: Set[int],
        var_positions2: Set[int]
    ) -> float:
        """Calculate similarity between token sequences."""
        # Get static token positions
        static1 = set(range(len(tokens1))) - var_positions1
        static2 = set(range(len(tokens2))) - var_positions2
        
        # If all tokens are variables, compare variable positions
        if not static1 and not static2:
            norm_vars1 = {i/len(tokens1) for i in var_positions1}
            norm_vars2 = {i/len(tokens2) for i in var_positions2}
            intersection = len(norm_vars1 & norm_vars2)
            union = len(norm_vars1 | norm_vars2)
            return intersection / union if union > 0 else 0.0
            
        # Compare static tokens
        static_tokens1 = [t for i, t in enumerate(tokens1) if i in static1]
        static_tokens2 = [t for i, t in enumerate(tokens2) if i in static2]
        
        # Use difflib sequence matcher ratio
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, static_tokens1, static_tokens2)
        return matcher.ratio()
    
    def match(self, template1: str, template2: str) -> Dict[str, Any]:
        """Match two templates and determine their similarity."""
        # Tokenize templates
        tokens1 = self._tokenize(template1)
        tokens2 = self._tokenize(template2)
        
        # Get variable positions
        var_positions1 = self._get_variable_positions(tokens1)
        var_positions2 = self._get_variable_positions(tokens2)
        
        # Check for exact match
        if template1 == template2:
            return {
                "match_type": "exact",
                "similarity_score": 1.0,
                "matched_positions": list(range(len(tokens1))),
                "variable_positions": list(var_positions1)
            }
            
        # Calculate similarity
        similarity = self._calculate_token_similarity(
            tokens1, tokens2,
            var_positions1, var_positions2
        )
        
        # Determine match type
        if similarity >= self.similarity_threshold:
            match_type = "similar"
        elif var_positions1 and var_positions2:
            match_type = "variable_only"
        else:
            match_type = "no_match"
            
        # Find matching positions
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, tokens1, tokens2)
        matched_positions = []
        for block in matcher.get_matching_blocks():
            matched_positions.extend(range(block.a, block.a + block.size))
            
        return {
            "match_type": match_type,
            "similarity_score": similarity,
            "matched_positions": matched_positions,
            "variable_positions": list(var_positions1)
        }


class NumericAnomalyDetector:
    """Python implementation of numeric anomaly detector for benchmarking."""
    
    def __init__(
        self,
        std_threshold: float = 3.0,
        iqr_threshold: float = 1.5,
        min_samples: int = 10,
        use_robust: bool = True
    ):
        self.std_threshold = std_threshold
        self.iqr_threshold = iqr_threshold
        self.min_samples = min_samples
        self.use_robust = use_robust
    
    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical summary for a numeric column."""
        values_arr = np.array(values)
        
        # Handle empty or near-empty arrays
        if len(values_arr) < 2:
            return {
                "mean": np.nan,
                "median": np.nan,
                "stddev": np.nan,
                "q1": np.nan,
                "q3": np.nan,
                "iqr": np.nan,
                "min": np.nan,
                "max": np.nan
            }
        
        # Calculate statistics
        q1, median, q3 = np.percentile(values_arr, [25, 50, 75])
        
        return {
            "mean": np.mean(values_arr),
            "median": median,
            "stddev": np.std(values_arr),
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
            "min": np.min(values_arr),
            "max": np.max(values_arr)
        }
    
    def detect_field_anomalies(
        self,
        values: List[float],
        stats: Dict[str, float]
    ) -> List[bool]:
        """Detect anomalies in a single numeric field."""
        if self.use_robust:
            # Use robust statistics (median/IQR)
            q1 = stats["q1"]
            q3 = stats["q3"]
            iqr = stats["iqr"]
            
            if iqr == 0:  # Handle zero IQR case
                return [False] * len(values)
                
            lower = q1 - (self.iqr_threshold * iqr)
            upper = q3 + (self.iqr_threshold * iqr)
            
            return [(v < lower or v > upper) for v in values]
        else:
            # Use classical statistics (mean/std)
            mean = stats["mean"]
            std = stats["stddev"]
            
            if std == 0:  # Handle zero std case
                return [False] * len(values)
                
            z_scores = np.abs((np.array(values) - mean) / std)
            return (z_scores > self.std_threshold).tolist()
    
    def detect_anomalies(
        self,
        data: List[List[float]],
        field_names: List[str],
        group_col: List[int] = None
    ) -> Dict[str, Any]:
        """Detect anomalies in numeric fields."""
        if not data or not field_names:
            return {
                "anomaly_mask": [],
                "field_anomalies": {},
                "field_stats": {}
            }
        
        # Convert data to numpy for easier processing
        data_arr = np.array(data)
        num_samples = data_arr.shape[0]
        num_fields = data_arr.shape[1]
        
        # Initialize results
        all_anomalies = np.zeros(num_samples, dtype=bool)
        field_anomalies = {}
        field_stats = {}
        
        # Process each field
        for i, field_name in enumerate(field_names):
            if i >= num_fields:
                continue
                
            values = data_arr[:, i]
            
            # Skip if not enough samples
            if len(values) < self.min_samples:
                field_anomalies[field_name] = [False] * num_samples
                continue
            
            # Group analysis
            if group_col is not None and len(group_col) == num_samples:
                anomaly_mask = np.zeros(num_samples, dtype=bool)
                
                # Process each group
                for group in set(group_col):
                    if group == -1:  # Skip noise points
                        continue
                        
                    group_mask = np.array(group_col) == group
                    group_values = values[group_mask]
                    
                    if len(group_values) >= self.min_samples:
                        stats = self.calculate_statistics(group_values.tolist())
                        field_stats[f"{field_name}_group_{group}"] = stats
                        
                        group_anomalies = self.detect_field_anomalies(
                            group_values.tolist(), stats
                        )
                        anomaly_mask[group_mask] = group_anomalies
            else:
                # Global analysis
                stats = self.calculate_statistics(values.tolist())
                field_stats[field_name] = stats
                
                anomaly_mask = self.detect_field_anomalies(values.tolist(), stats)
            
            field_anomalies[field_name] = anomaly_mask
            all_anomalies = np.logical_or(all_anomalies, anomaly_mask)
        
        return {
            "anomaly_mask": all_anomalies.tolist(),
            "field_anomalies": field_anomalies,
            "field_stats": field_stats
        }


def benchmark_template_matcher(iterations: int) -> Dict[str, Any]:
    """Benchmark the template matcher."""
    # Create test templates
    template_pairs = [
        ("User <OID> logged in from <LOI>", "User admin logged in from 192.168.1.1"),
        ("Failed to connect to server <OBN> after <OBA> attempts", "Failed to connect to server db01 after 3 attempts"),
        ("Process <PID> started with arguments <OTP>", "Process 1234 started with arguments --verbose")
    ]
    
    matcher = TemplateMatcher(0.8)
    
    start_time = time.time()
    
    for _ in range(iterations):
        for template1, template2 in template_pairs:
            result = matcher.match(template1, template2)
            # Prevent optimization
            if result["similarity_score"] < 0:
                print("Error")
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    
    return {
        "name": "Template Matcher",
        "avg_time_ms": total_time_ms / (iterations * len(template_pairs)),
        "throughput": (iterations * len(template_pairs) * 1000) / total_time_ms,
        "iterations": iterations * len(template_pairs)
    }


def benchmark_numeric_analysis(iterations: int) -> Dict[str, Any]:
    """Benchmark the numeric anomaly detector."""
    # Create test data
    data = []
    field_names = ["value1", "value2", "value3"]
    
    # Generate random data with some anomalies
    np.random.seed(42)
    
    for i in range(10000):
        row = []
        for j in range(3):
            value = np.random.normal(100.0, 15.0)
            if i % 100 == 0:
                # Add anomalies
                value *= 3
            row.append(value)
        data.append(row)
    
    detector = NumericAnomalyDetector(3.0, 1.5, 10, True)
    
    start_time = time.time()
    
    for _ in range(iterations):
        result = detector.detect_anomalies(data, field_names)
        # Prevent optimization
        if not result["anomaly_mask"]:
            print("Error")
    
    end_time = time.time()
    total_time_ms = (end_time - start_time) * 1000
    
    return {
        "name": "Numeric Analysis",
        "avg_time_ms": total_time_ms / iterations,
        "throughput": (iterations * 1000) / total_time_ms,
        "iterations": iterations
    }


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python benchmark_python.py <benchmark_name>")
        sys.exit(1)
    
    benchmark_name = sys.argv[1]
    
    if benchmark_name == "template_matcher":
        result = benchmark_template_matcher(100)
    elif benchmark_name == "numeric_analysis":
        result = benchmark_numeric_analysis(5)
    else:
        print(f"Unknown benchmark: {benchmark_name}")
        sys.exit(1)
    
    # Print result in a format that can be parsed by the C++ program
    print(f"name:{result['name']}")
    print(f"avg_time_ms:{result['avg_time_ms']}")
    print(f"throughput:{result['throughput']}")
    print(f"iterations:{result['iterations']}")


if __name__ == "__main__":
    main()