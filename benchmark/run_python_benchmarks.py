#!/usr/bin/env python3
"""Run benchmarks on the Python implementation."""

import time
import json
import os
import sys
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.numeric_analysis import NumericAnomalyDetector
from src.models.log_parser import LogParser

# Setup output directory
REPORT_DIR = "benchmark_reports"
os.makedirs(REPORT_DIR, exist_ok=True)

class Timer:
    """Simple timer for benchmarking."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        print(f"{self.name}: {self.duration:.6f} seconds")


def generate_random_data(num_samples: int, num_fields: int) -> pd.DataFrame:
    """Generate random data for benchmarking."""
    data = {}
    
    # Generate random values
    for i in range(num_fields):
        data[f"field_{i}"] = np.random.uniform(0, 100, num_samples)
        
    # Add some anomalies (5%)
    num_anomalies = int(num_samples * 0.05)
    for i in range(num_anomalies):
        field_idx = np.random.randint(0, num_fields)
        sample_idx = np.random.randint(0, num_samples)
        data[f"field_{field_idx}"][sample_idx] = np.random.uniform(200, 500)
        
    return pd.DataFrame(data)


def generate_group_ids(num_samples: int, num_groups: int, grouped_ratio: float = 0.3) -> List[int]:
    """Generate random group IDs."""
    group_ids = [-1] * num_samples
    num_grouped = int(num_samples * grouped_ratio)
    
    for i in range(num_grouped):
        idx = np.random.randint(0, num_samples)
        group_ids[idx] = np.random.randint(0, num_groups)
        
    return group_ids


def benchmark_detect_field_anomalies(sizes: List[int]) -> List[Dict[str, Any]]:
    """Benchmark detectFieldAnomalies method."""
    results = []
    
    for size in sizes:
        # Generate data
        values = np.random.uniform(0, 100, size)
        
        # Add anomalies (5%)
        num_anomalies = int(size * 0.05)
        for i in range(num_anomalies):
            idx = np.random.randint(0, size)
            values[idx] = np.random.uniform(200, 500)
            
        # Create detector
        detector = NumericAnomalyDetector()
        
        # Warm-up
        _ = detector._detect_field_anomalies(pd.Series(values))
        
        # Benchmark
        iterations = max(1, int(10000 / size))  # Reduce iterations for large sizes
        total_time = 0
        
        for _ in range(iterations):
            with Timer(f"detect_field_anomalies({size})") as t:
                _ = detector._detect_field_anomalies(pd.Series(values))
            total_time += t.duration
            
        avg_time = total_time / iterations
        
        results.append({
            "name": f"BM_DetectFieldAnomalies/{size}",
            "iterations": iterations,
            "real_time": avg_time * 1e9,  # Convert to nanoseconds
            "cpu_time": avg_time * 1e9,
            "time_unit": "ns",
            "items_per_second": size / avg_time
        })
        
    return results


def benchmark_detect_anomalies(configs: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    """Benchmark detectAnomalies method."""
    results = []
    
    for config in configs:
        num_samples = config["num_samples"]
        num_fields = config["num_fields"]
        
        # Generate data
        df = generate_random_data(num_samples, num_fields)
        group_ids = generate_group_ids(num_samples, 5)
        
        # Create detector
        detector = NumericAnomalyDetector()
        
        # Warm-up
        _ = detector.detect_anomalies(df)
        
        # Benchmark
        iterations = max(1, int(1000 / (num_samples * num_fields)))  # Reduce iterations for large sizes
        total_time = 0
        
        for _ in range(iterations):
            with Timer(f"detect_anomalies({num_samples}, {num_fields})") as t:
                _ = detector.detect_anomalies(df, group_by="group")
            total_time += t.duration
            
        avg_time = total_time / iterations
        
        results.append({
            "name": f"BM_DetectAnomalies/{num_samples}_{num_fields}",
            "iterations": iterations,
            "real_time": avg_time * 1e9,  # Convert to nanoseconds
            "cpu_time": avg_time * 1e9,
            "time_unit": "ns",
            "items_per_second": (num_samples * num_fields) / avg_time
        })
        
    return results


def benchmark_extract_numeric_fields(configs: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    """Benchmark extractNumericFields method."""
    results = []
    
    for config in configs:
        num_samples = config["num_samples"]
        num_fields = config["num_fields"]
        
        # Generate data with mixed types
        df = pd.DataFrame()
        
        # Add numeric fields (70%)
        num_numeric = int(num_fields * 0.7)
        for i in range(num_numeric):
            df[f"numeric_{i}"] = np.random.uniform(0, 100, num_samples)
            
        # Add string fields (30%)
        for i in range(num_numeric, num_fields):
            df[f"string_{i}"] = [f"string-{j}" for j in range(num_samples)]
            
        # Create detector
        detector = NumericAnomalyDetector()
        
        # Warm-up
        _ = detector._extract_numeric_fields(df)
        
        # Benchmark
        iterations = max(1, int(1000 / (num_samples * num_fields)))  # Reduce iterations for large sizes
        total_time = 0
        
        for _ in range(iterations):
            with Timer(f"extract_numeric_fields({num_samples}, {num_fields})") as t:
                _ = detector._extract_numeric_fields(df)
            total_time += t.duration
            
        avg_time = total_time / iterations
        
        results.append({
            "name": f"BM_ExtractNumericFields/{num_samples}_{num_fields}",
            "iterations": iterations,
            "real_time": avg_time * 1e9,  # Convert to nanoseconds
            "cpu_time": avg_time * 1e9,
            "time_unit": "ns",
            "items_per_second": (num_samples * num_fields) / avg_time
        })
        
    return results


def generate_random_logs(num_logs: int) -> List[str]:
    """Generate random log messages for benchmarking."""
    templates = [
        "Connection to {server} failed after {retry} attempts",
        "User {username} logged in from {ip_address}",
        "Database query took {time}ms to execute",
        "Memory usage at {percent}%, system load: {load}",
        "Failed to process request from {client}: {error_code}",
        "Thread {thread_id} waiting for lock on resource {resource_id}",
        "API request to {endpoint} returned status {status_code}",
        "File {filename} not found in directory {directory}",
        "Successfully processed {count} items in {duration}s",
        "Cache hit ratio: {ratio}%, misses: {misses}"
    ]
    
    log_messages = []
    rng = np.random.RandomState(42)
    
    for _ in range(num_logs):
        template = templates[rng.randint(0, len(templates))]
        
        # Replace placeholders with random values
        log = template
        if "{server}" in log:
            log = log.replace("{server}", f"server-{rng.randint(1, 100)}")
        if "{retry}" in log:
            log = log.replace("{retry}", str(rng.randint(1, 10)))
        if "{username}" in log:
            log = log.replace("{username}", f"user{rng.randint(1000, 9999)}")
        if "{ip_address}" in log:
            log = log.replace("{ip_address}", f"{rng.randint(1, 255)}.{rng.randint(1, 255)}.{rng.randint(1, 255)}.{rng.randint(1, 255)}")
        if "{time}" in log:
            log = log.replace("{time}", str(rng.randint(1, 5000)))
        if "{percent}" in log:
            log = log.replace("{percent}", str(rng.randint(1, 100)))
        if "{load}" in log:
            log = log.replace("{load}", f"{rng.uniform(0.1, 10.0):.2f}")
        if "{client}" in log:
            log = log.replace("{client}", f"client-{rng.randint(1, 1000)}")
        if "{error_code}" in log:
            log = log.replace("{error_code}", f"ERR{rng.randint(100, 999)}")
        if "{thread_id}" in log:
            log = log.replace("{thread_id}", f"thread-{rng.randint(1, 100)}")
        if "{resource_id}" in log:
            log = log.replace("{resource_id}", f"res-{rng.randint(1, 50)}")
        if "{endpoint}" in log:
            log = log.replace("{endpoint}", f"/api/v{rng.randint(1, 3)}/resources/{rng.randint(1, 1000)}")
        if "{status_code}" in log:
            log = log.replace("{status_code}", str(rng.choice([200, 201, 400, 401, 403, 404, 500])))
        if "{filename}" in log:
            log = log.replace("{filename}", f"file{rng.randint(1, 1000)}.txt")
        if "{directory}" in log:
            log = log.replace("{directory}", f"/var/data/dir{rng.randint(1, 50)}")
        if "{count}" in log:
            log = log.replace("{count}", str(rng.randint(1, 10000)))
        if "{duration}" in log:
            log = log.replace("{duration}", f"{rng.uniform(0.1, 60.0):.2f}")
        if "{ratio}" in log:
            log = log.replace("{ratio}", f"{rng.uniform(10.0, 99.9):.1f}")
        if "{misses}" in log:
            log = log.replace("{misses}", str(rng.randint(1, 1000)))
            
        log_messages.append(log)
    
    return log_messages

def benchmark_log_parser(sizes: List[int]) -> List[Dict[str, Any]]:
    """Benchmark log parser performance."""
    results = []
    
    for size in sizes:
        # Generate random logs
        logs = generate_random_logs(size)
        
        try:
            # Initialize parser
            parser = LogParser()
            
            # Warm-up
            _ = parser.parse_logs(logs[:min(100, len(logs))])
            
            # Benchmark parsing
            iterations = max(1, int(1000 / size))  # Fewer iterations for larger inputs
            total_time = 0
            
            for _ in range(iterations):
                with Timer(f"parse_logs({size})") as t:
                    _ = parser.parse_logs(logs)
                total_time += t.duration
                
            avg_time = total_time / iterations
            
            results.append({
                "name": f"BM_LogParser/{size}",
                "iterations": iterations,
                "real_time": avg_time * 1e9,  # Convert to nanoseconds
                "cpu_time": avg_time * 1e9,
                "time_unit": "ns",
                "items_per_second": size / avg_time
            })
            
        except Exception as e:
            print(f"Error benchmarking LogParser with {size} logs: {e}")
    
    return results

def main():
    """Run all benchmarks and save results."""
    all_results = []
    
    # Log parsing benchmark
    print("\n=== Benchmarking LogParser ===")
    sizes = [10, 100, 1000, 5000]
    results = benchmark_log_parser(sizes)
    all_results.extend(results)
    
    # Detect field anomalies benchmark
    print("\n=== Benchmarking DetectFieldAnomalies ===")
    sizes = [100, 1000, 10000, 100000, 1000000]
    results = benchmark_detect_field_anomalies(sizes)
    all_results.extend(results)
    
    # Detect anomalies benchmark
    print("\n=== Benchmarking DetectAnomalies ===")
    configs = [
        {"num_samples": 1000, "num_fields": 10},
        {"num_samples": 10000, "num_fields": 10},
        {"num_samples": 100000, "num_fields": 10},
        {"num_samples": 1000, "num_fields": 50},
        {"num_samples": 10000, "num_fields": 50}
    ]
    results = benchmark_detect_anomalies(configs)
    all_results.extend(results)
    
    # Extract numeric fields benchmark
    print("\n=== Benchmarking ExtractNumericFields ===")
    configs = [
        {"num_samples": 100, "num_fields": 10},
        {"num_samples": 1000, "num_fields": 10},
        {"num_samples": 10000, "num_fields": 10},
        {"num_samples": 1000, "num_fields": 50}
    ]
    results = benchmark_extract_numeric_fields(configs)
    all_results.extend(results)
    
    # Save results
    benchmark_result = {"context": {}, "benchmarks": all_results}
    
    output_file = os.path.join(REPORT_DIR, "python_benchmark.json")
    with open(output_file, "w") as f:
        json.dump(benchmark_result, f, indent=2)
        
    print(f"\nBenchmark results saved to {output_file}")


if __name__ == "__main__":
    main()