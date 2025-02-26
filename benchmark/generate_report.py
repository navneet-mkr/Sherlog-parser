#!/usr/bin/env python3
"""Generate benchmark comparison report."""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Constants
REPORT_DIR = "benchmark_reports"
CPP_BENCHMARK_FILE = os.path.join(REPORT_DIR, "cpp_numeric_benchmark.json")
PYTHON_BENCHMARK_FILE = os.path.join(REPORT_DIR, "python_numeric_benchmark.json")
OUTPUT_HTML = os.path.join(REPORT_DIR, "benchmark_report.html")
OUTPUT_CSV = os.path.join(REPORT_DIR, "benchmark_comparison.csv")


def load_benchmark_data(file_path):
    """Load benchmark data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def parse_benchmark_name(name):
    """Parse benchmark name to extract function name and parameters."""
    parts = name.split('/')
    func_name = parts[0]
    
    # Parse parameters
    params = {}
    if len(parts) > 1:
        param_str = parts[1]
        if '_' in param_str:
            # Format: 1000_10 (num_samples_num_fields)
            param_parts = param_str.split('_')
            params['num_samples'] = int(param_parts[0])
            params['num_fields'] = int(param_parts[1])
        else:
            # Format: 1000 (size)
            params['size'] = int(param_str)
    
    return func_name, params


def create_comparison_dataframe(cpp_data, python_data):
    """Create a dataframe comparing C++ and Python benchmark results."""
    rows = []
    
    # Process C++ benchmarks
    for bench in cpp_data['benchmarks']:
        name = bench['name']
        func_name, params = parse_benchmark_name(name)
        
        row = {
            'function': func_name,
            'params': str(params),
            'cpp_time_ns': bench['real_time'],
            'cpp_items_per_second': bench.get('items_processed', 0) / (bench['real_time'] / 1e9)
        }
        
        # Add specific parameters for filtering
        for key, value in params.items():
            row[key] = value
            
        rows.append(row)
    
    # Process Python benchmarks
    for bench in python_data['benchmarks']:
        name = bench['name']
        func_name, params = parse_benchmark_name(name)
        
        # Find matching C++ benchmark
        for row in rows:
            if row['function'] == func_name and str(params) == row['params']:
                row['python_time_ns'] = bench['real_time']
                row['python_items_per_second'] = bench.get('items_per_second', 0)
                break
    
    # Calculate speedup ratio
    for row in rows:
        if 'python_time_ns' in row and 'cpp_time_ns' in row:
            row['speedup'] = row['python_time_ns'] / row['cpp_time_ns']
    
    return pd.DataFrame(rows)


def create_comparison_plots(df):
    """Create comparison plots."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Execution Time Comparison",
            "Speedup Ratio (higher is better)",
            "Items Processed per Second",
            "Detailed Comparison"
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {"type": "table"}]
        ]
    )
    
    # Filter functions
    functions = df['function'].unique()
    
    # Plot 1: Execution Time Comparison (log scale)
    for func in functions:
        func_df = df[df['function'] == func]
        
        # For functions with size parameter
        if 'size' in func_df.columns and not func_df['size'].isna().all():
            sizes = sorted(func_df['size'].unique())
            cpp_times = [func_df[func_df['size'] == size]['cpp_time_ns'].values[0] / 1e6 for size in sizes]
            python_times = [func_df[func_df['size'] == size]['python_time_ns'].values[0] / 1e6 for size in sizes]
            
            fig.add_trace(
                go.Bar(
                    x=[f"{func} (n={size})" for size in sizes],
                    y=cpp_times,
                    name=f"{func} (C++)",
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=[f"{func} (n={size})" for size in sizes],
                    y=python_times,
                    name=f"{func} (Python)",
                    marker_color='red'
                ),
                row=1, col=1
            )
        
        # For functions with num_samples and num_fields
        elif 'num_samples' in func_df.columns and 'num_fields' in func_df.columns:
            configs = [(row['num_samples'], row['num_fields']) for _, row in func_df.iterrows()]
            configs = sorted(configs)
            
            cpp_times = []
            python_times = []
            labels = []
            
            for samples, fields in configs:
                mask = (func_df['num_samples'] == samples) & (func_df['num_fields'] == fields)
                if not mask.any():
                    continue
                    
                row = func_df[mask].iloc[0]
                cpp_times.append(row['cpp_time_ns'] / 1e6)
                python_times.append(row['python_time_ns'] / 1e6)
                labels.append(f"{func} ({samples}x{fields})")
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=cpp_times,
                    name=f"{func} (C++)",
                    marker_color='blue'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=labels,
                    y=python_times,
                    name=f"{func} (Python)",
                    marker_color='red'
                ),
                row=1, col=1
            )
    
    # Plot 2: Speedup Ratio
    x_labels = []
    speedups = []
    
    for _, row in df.iterrows():
        if 'speedup' in row and not np.isnan(row['speedup']):
            func = row['function']
            if 'size' in row and not pd.isna(row['size']):
                label = f"{func} (n={int(row['size'])})"
            elif 'num_samples' in row and 'num_fields' in row:
                label = f"{func} ({int(row['num_samples'])}x{int(row['num_fields'])})"
            else:
                label = func
                
            x_labels.append(label)
            speedups.append(row['speedup'])
    
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=speedups,
            name="Speedup Ratio (Python/C++)",
            marker_color='green'
        ),
        row=1, col=2
    )
    
    # Plot 3: Items Processed per Second
    for func in functions:
        func_df = df[df['function'] == func]
        
        x_labels = []
        cpp_throughput = []
        python_throughput = []
        
        for _, row in func_df.iterrows():
            if 'size' in row and not pd.isna(row['size']):
                label = f"{func} (n={int(row['size'])})"
            elif 'num_samples' in row and 'num_fields' in row:
                label = f"{func} ({int(row['num_samples'])}x{int(row['num_fields'])})"
            else:
                label = func
                
            x_labels.append(label)
            cpp_throughput.append(row['cpp_items_per_second'])
            python_throughput.append(row['python_items_per_second'])
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=cpp_throughput,
                name=f"{func} Throughput (C++)",
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(
                x=x_labels,
                y=python_throughput,
                name=f"{func} Throughput (Python)",
                marker_color='salmon'
            ),
            row=2, col=1
        )
    
    # Plot 4: Table with detailed results
    table_df = df.copy()
    table_df['cpp_time_ms'] = table_df['cpp_time_ns'] / 1e6
    table_df['python_time_ms'] = table_df['python_time_ns'] / 1e6
    
    # Format the table
    table_data = []
    
    # Headers
    table_headers = ["Function", "Parameters", "C++ Time (ms)", "Python Time (ms)", "Speedup"]
    table_data.append(table_headers)
    
    # Rows
    for _, row in table_df.iterrows():
        if 'size' in row and not pd.isna(row['size']):
            params = f"size={int(row['size'])}"
        elif 'num_samples' in row and 'num_fields' in row:
            params = f"samples={int(row['num_samples'])}, fields={int(row['num_fields'])}"
        else:
            params = row['params']
            
        table_row = [
            row['function'],
            params,
            f"{row['cpp_time_ms']:.2f}",
            f"{row['python_time_ms']:.2f}",
            f"{row['speedup']:.2f}x"
        ]
        table_data.append(table_row)
    
    fig.add_trace(
        go.Table(
            header=dict(
                values=table_headers,
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=list(zip(*table_data[1:])),
                fill_color='lavender',
                align='left'
            )
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text="C++ vs Python Implementation Performance Comparison",
        barmode='group',
        height=1000,
        width=1200
    )
    
    fig.update_yaxes(type="log", title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Speedup (x times)", row=1, col=2)
    fig.update_yaxes(type="log", title_text="Items/second", row=2, col=1)
    
    return fig


def main():
    """Generate benchmark comparison report."""
    try:
        print(f"Loading C++ benchmark data from {CPP_BENCHMARK_FILE}")
        cpp_data = load_benchmark_data(CPP_BENCHMARK_FILE)
        
        print(f"Loading Python benchmark data from {PYTHON_BENCHMARK_FILE}")
        python_data = load_benchmark_data(PYTHON_BENCHMARK_FILE)
        
        print("Creating comparison dataframe")
        df = create_comparison_dataframe(cpp_data, python_data)
        
        # Save comparison data
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved comparison data to {OUTPUT_CSV}")
        
        print("Creating comparison plots")
        fig = create_comparison_plots(df)
        
        # Save HTML report
        fig.write_html(OUTPUT_HTML)
        print(f"Saved HTML report to {OUTPUT_HTML}")
        
        # Also save a static image
        fig.write_image(os.path.join(REPORT_DIR, "benchmark_comparison.png"))
        print(f"Saved static image to {os.path.join(REPORT_DIR, 'benchmark_comparison.png')}")
        
        print("Report generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        raise


if __name__ == "__main__":
    main()