#!/bin/bash
set -e

# Create build directories
mkdir -p build bin benchmark_reports

# Check if Google Benchmark is installed
if ! pkg-config --exists benchmark; then
    echo "Google Benchmark not found. Installing..."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        brew install google-benchmark
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update
        sudo apt-get install -y libgoogle-benchmark-dev
    else
        echo "Unsupported OS. Please install Google Benchmark manually."
        exit 1
    fi
fi

# Build benchmarks
echo "Building benchmarks..."
make numeric_benchmark log_parser_benchmark

# Run the numeric analysis benchmark
echo "Running numeric analysis benchmark..."
./bin/numeric_benchmark --benchmark_out=benchmark_reports/cpp_numeric_benchmark.json --benchmark_out_format=json

# Run the log parser benchmark
echo "Running log parser benchmark..."
./bin/log_parser_benchmark --benchmark_out=benchmark_reports/cpp_log_parser_benchmark.json --benchmark_out_format=json

# Run the Python benchmarks
echo "Running Python benchmarks..."
cd ..
mkdir -p benchmark/benchmark_reports
python benchmark/run_python_benchmarks.py

# Compare results
echo "Generating comparison reports..."
python -c "
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

plt.style.use('ggplot')

# Create output directory
os.makedirs('benchmark_reports', exist_ok=True)

# Load benchmark results
try:
    with open('benchmark/benchmark_reports/python_benchmark.json', 'r') as f:
        py_results = json.load(f)
    
    with open('cpp/benchmark_reports/cpp_numeric_benchmark.json', 'r') as f:
        cpp_numeric_results = json.load(f)
        
    with open('cpp/benchmark_reports/cpp_log_parser_benchmark.json', 'r') as f:
        cpp_log_parser_results = json.load(f)
except FileNotFoundError as e:
    print(f'Error: {e}')
    sys.exit(1)

# Process results
py_data = []
for bench in py_results['benchmarks']:
    name = bench['name']
    time_ns = bench['real_time']
    items_per_sec = bench.get('items_per_second', 0)
    py_data.append({
        'name': name,
        'language': 'Python',
        'time_ns': time_ns,
        'items_per_sec': items_per_sec
    })

cpp_data = []
for bench in cpp_numeric_results['benchmarks'] + cpp_log_parser_results['benchmarks']:
    name = bench['name']
    time_ns = bench['real_time']
    items_per_sec = bench.get('items_per_second', 0)
    cpp_data.append({
        'name': name,
        'language': 'C++',
        'time_ns': time_ns,
        'items_per_sec': items_per_sec
    })

# Create DataFrame
all_data = pd.DataFrame(py_data + cpp_data)

# Generate comparative plots
numeric_benchmarks = [
    'BM_DetectFieldAnomalies',
    'BM_DetectAnomalies',
    'BM_ExtractNumericFields'
]

parsing_benchmarks = [
    'BM_LogParser',
    'BM_ParseSingleLog',
    'BM_ParseLogBatch'
]

# Function to generate plots
def create_comparison_plot(df, benchmark_prefix, title, filename):
    filtered_df = df[df['name'].str.startswith(benchmark_prefix)]
    if filtered_df.empty:
        print(f'No data for {benchmark_prefix}')
        return
        
    # Get size parameter from benchmark name
    filtered_df['size'] = filtered_df['name'].str.extract(r'/(\d+)').astype(float)
    filtered_df = filtered_df.sort_values('size')
    
    # Create plot
    plt.figure(figsize=(10, 6))
    for lang in filtered_df['language'].unique():
        lang_df = filtered_df[filtered_df['language'] == lang]
        plt.plot(lang_df['size'], lang_df['time_ns'] / 1e6, 'o-', label=lang)
    
    plt.title(title)
    plt.xlabel('Input Size')
    plt.ylabel('Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'benchmark_reports/{filename}.png')
    
    # Create speedup table
    if 'Python' in filtered_df['language'].values and 'C++' in filtered_df['language'].values:
        py_df = filtered_df[filtered_df['language'] == 'Python']
        cpp_df = filtered_df[filtered_df['language'] == 'C++']
        
        sizes = sorted(set(py_df['size'].unique()) & set(cpp_df['size'].unique()))
        
        speedup_data = []
        for size in sizes:
            py_time = py_df[py_df['size'] == size]['time_ns'].values[0]
            cpp_time = cpp_df[cpp_df['size'] == size]['time_ns'].values[0]
            speedup = py_time / cpp_time
            speedup_data.append({
                'Input Size': size,
                'Python Time (ms)': py_time / 1e6,
                'C++ Time (ms)': cpp_time / 1e6,
                'Speedup Factor': speedup
            })
        
        speedup_df = pd.DataFrame(speedup_data)
        print(f'\\n{title} - Speedup Results:')
        print(speedup_df.to_string(index=False))
        
        # Save to CSV
        speedup_df.to_csv(f'benchmark_reports/{filename}_speedup.csv', index=False)

# Generate plots for each benchmark type
for prefix in numeric_benchmarks:
    create_comparison_plot(all_data, prefix, f'{prefix} Benchmark', f'{prefix.lower()}_comparison')

for prefix in parsing_benchmarks:
    create_comparison_plot(all_data, prefix, f'{prefix} Benchmark', f'{prefix.lower()}_comparison')

print('\\nBenchmark comparison complete. Results saved to benchmark_reports directory.')
"

echo "Benchmarking complete! Results are in the benchmark_reports directory."