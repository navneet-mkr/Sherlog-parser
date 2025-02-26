#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create HTML report file
report_file="performance_comparison_report.html"

# Run benchmark script first
./run_benchmark.sh | tee benchmark_results.txt

# Extract results for report
template_cpp_time=$(grep "Template Matcher" benchmark_results.txt | head -1 | awk '{print $3}')
template_cpp_throughput=$(grep "Template Matcher" benchmark_results.txt | head -1 | awk '{print $4}')
numeric_cpp_time=$(grep "Numeric Analysis" benchmark_results.txt | head -1 | awk '{print $3}')
numeric_cpp_throughput=$(grep "Numeric Analysis" benchmark_results.txt | head -1 | awk '{print $4}')

template_py_time=$(grep "Python Template Matcher" benchmark_results.txt | awk '{print $4}')
template_py_throughput=$(grep "Python Template Matcher" benchmark_results.txt | awk '{print $5}')
numeric_py_time=$(grep "Python Numeric Analysis" benchmark_results.txt | awk '{print $4}')
numeric_py_throughput=$(grep "Python Numeric Analysis" benchmark_results.txt | awk '{print $5}')

template_speedup=$(grep "Template Matcher" benchmark_results.txt | tail -2 | head -1 | awk '{print $4}')
numeric_speedup=$(grep "Numeric Analysis" benchmark_results.txt | tail -1 | awk '{print $4}')

# Create HTML report
cat > $report_file << EOL
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sherlog C++ vs Python Performance Benchmark</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .highlight {
            background-color: #e74c3c;
            color: white;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 3px;
        }
        .summary {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 5px solid #3498db;
            margin-bottom: 20px;
        }
        .chart-container {
            height: 400px;
            margin-bottom: 30px;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Sherlog C++ vs Python Performance Benchmark</h1>
    
    <div class="summary">
        <h2>Performance Summary</h2>
        <p>The C++ implementation provides:</p>
        <ul>
            <li>Template Matcher: <strong>${template_speedup}x speedup</strong> over Python</li>
            <li>Numeric Analysis: <strong>${numeric_speedup}x speedup</strong> over Python</li>
        </ul>
    </div>
    
    <h2>Detailed Metrics</h2>
    
    <table>
        <tr>
            <th>Component</th>
            <th>Implementation</th>
            <th>Avg Time (ms)</th>
            <th>Throughput (items/s)</th>
            <th>Speedup</th>
        </tr>
        <tr>
            <td rowspan="2">Template Matcher</td>
            <td>C++</td>
            <td>${template_cpp_time}</td>
            <td>${template_cpp_throughput}</td>
            <td rowspan="2">${template_speedup}x</td>
        </tr>
        <tr>
            <td>Python</td>
            <td>${template_py_time}</td>
            <td>${template_py_throughput}</td>
        </tr>
        <tr>
            <td rowspan="2">Numeric Analysis</td>
            <td>C++</td>
            <td>${numeric_cpp_time}</td>
            <td>${numeric_cpp_throughput}</td>
            <td rowspan="2">${numeric_speedup}x</td>
        </tr>
        <tr>
            <td>Python</td>
            <td>${numeric_py_time}</td>
            <td>${numeric_py_throughput}</td>
        </tr>
    </table>
    
    <h2>Performance Visualization</h2>
    
    <div class="chart-container">
        <canvas id="timeChart"></canvas>
    </div>
    
    <div class="chart-container">
        <canvas id="throughputChart"></canvas>
    </div>
    
    <h2>Conclusion</h2>
    <p>
        The C++ implementation demonstrates significant performance advantages over the Python version.
        For template matching operations, C++ is <strong>${template_speedup}x faster</strong>, while numeric analysis
        achieves a <strong>${numeric_speedup}x speedup</strong>.
    </p>
    <p>
        These improvements are particularly important for high-volume log processing scenarios, where
        the performance gains can translate to substantial time savings and resource efficiency in production
        environments.
    </p>
    
    <script>
        // Execution time chart
        const timeCtx = document.getElementById('timeChart').getContext('2d');
        const timeChart = new Chart(timeCtx, {
            type: 'bar',
            data: {
                labels: ['Template Matcher', 'Numeric Analysis'],
                datasets: [
                    {
                        label: 'C++ (ms)',
                        data: [${template_cpp_time}, ${numeric_cpp_time}],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Python (ms)',
                        data: [${template_py_time}, ${numeric_py_time}],
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgba(231, 76, 60, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Execution Time (ms)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Component'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Execution Time Comparison (lower is better)'
                    }
                }
            }
        });
        
        // Throughput chart
        const throughputCtx = document.getElementById('throughputChart').getContext('2d');
        const throughputChart = new Chart(throughputCtx, {
            type: 'bar',
            data: {
                labels: ['Template Matcher', 'Numeric Analysis'],
                datasets: [
                    {
                        label: 'C++ (items/s)',
                        data: [${template_cpp_throughput}, ${numeric_cpp_throughput}],
                        backgroundColor: 'rgba(52, 152, 219, 0.7)',
                        borderColor: 'rgba(52, 152, 219, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Python (items/s)',
                        data: [${template_py_throughput}, ${numeric_py_throughput}],
                        backgroundColor: 'rgba(231, 76, 60, 0.7)',
                        borderColor: 'rgba(231, 76, 60, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Throughput (items/s)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Component'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Throughput Comparison (higher is better)'
                    }
                }
            }
        });
    </script>
</body>
</html>
EOL

echo -e "${GREEN}Performance report generated: ${CYAN}$report_file${NC}"
echo -e "Open this file in a web browser to view the detailed report with charts."