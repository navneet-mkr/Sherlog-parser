#!/usr/bin/env python3

# Standard library imports
import argparse
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import subprocess
from dataclasses import asdict

# Configure logging with timestamp and level
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

def check_dependencies():
    """Check if all required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are installed, False otherwise
    """
    try:
        import numpy
        import pandas
        import sklearn
        import sentence_transformers
        import ollama
        import streamlit
        import plotly
        return True
    except ImportError as e:
        console.print(f"[red]Missing dependency: {str(e)}[/red]")
        console.print("\nPlease install all required dependencies:")
        console.print("pip install -r requirements.txt")
        return False

def check_ollama():
    """Check if Ollama service is running and has the required Qwen2.5-Coder model.
    
    Returns:
        bool: True if Ollama is running and model is available, False otherwise
    """
    import requests
    try:
        # Check if Ollama API is accessible
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise ConnectionError("Ollama is not running")
        
        # Check if required model exists, pull if not found
        tags = response.json()
        if not any(tag.get('name') == 'qwen2.5-coder' for tag in tags.get('models', [])):
            console.print("[yellow]Qwen2.5-Coder model not found. Pulling...[/yellow]")
            requests.post("http://localhost:11434/api/pull", json={"name": "qwen2.5-coder"})
        return True
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {str(e)}[/red]")
        console.print("\nPlease ensure Ollama is running:")
        console.print("1. Install Ollama from https://ollama.ai")
        console.print("2. Start Ollama service")
        console.print("3. Pull the Qwen2.5-Coder model: ollama pull qwen2.5-coder")
        return False

def check_datasets():
    """Check if evaluation datasets are present in the expected directory.
    
    Returns:
        bool: True if datasets exist, False otherwise
    """
    dataset_path = Path("./data/eval_datasets")
    if not dataset_path.exists():
        console.print("[yellow]Evaluation datasets not found[/yellow]")
        console.print("\nPlease download the datasets:")
        console.print("1. Create data directory: mkdir -p data/eval_datasets")
        console.print("2. Download and extract datasets into data/eval_datasets")
        return False
    return True

def create_directories():
    """Create necessary directory structure for evaluation outputs, cache, and logs."""
    directories = [
        "data/eval_datasets",  # For input datasets
        "output/eval",         # For evaluation results
        "cache/eval",          # For caching intermediate results
        "logs"                 # For log files
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def run_evaluation(args):
    """Run the log parsing evaluation process with progress tracking.
    
    Args:
        args: Command line arguments containing evaluation configuration
    """
    from src.core.eval import Evaluator
    
    # Initialize progress bar with custom styling
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow][[/bold yellow][progress.description]{task.description}[bold yellow]][/bold yellow]"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold yellow]•[/bold yellow]"),
        TextColumn("[bold blue]{task.fields[status]}"),
        transient=True,
    ) as progress:
        # Create main progress tracking task
        main_task = progress.add_task(
            "[bold magenta]Evaluation Progress[/bold magenta]",
            total=100,
            status="Initializing..."
        )
        
        # Initialize evaluator with configuration
        progress.update(main_task, status="[bold green]Loading evaluator...[/bold green]")
        evaluator = Evaluator(
            base_dir="./data/eval_datasets",
            dataset_type=args.dataset_type,
            system=args.system,
            llm_model="qwen2.5-coder",
            llm_api_base=f"http://localhost:{args.ollama_port}",
            output_dir="./output/eval",
            cache_dir="./cache/eval",
            track_api_calls=True  # Enable API call tracking
        )
        progress.update(main_task, completed=10, status="[bold green]Evaluator loaded[/bold green]")
        
        # Run evaluation process
        progress.update(main_task, completed=15, status="[bold yellow]Starting evaluation...[/bold yellow]")
        metrics = evaluator.evaluate()
        
        # Save evaluation results
        progress.update(main_task, completed=90, status="[bold yellow]Saving results...[/bold yellow]")
        
        # Write metrics to JSON file
        dataset_name = f"{args.system}_{args.dataset_type}"
        output_dir = Path("./output/eval")
        with open(output_dir / f"{dataset_name}_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Mark completion
        progress.update(main_task, completed=100, status="[bold green]Complete![/bold green]")
        
        # Display results with rich formatting
        console.print("\n[bold green]✨ Evaluation complete! ✨[/bold green]")
        console.print("\n[bold magenta]📊 Evaluation Results[/bold magenta]")
        console.print("[yellow]" + "=" * 50 + "[/yellow]")
        
        # Convert metrics to dictionary and group by category
        metrics_dict = asdict(metrics)
        
        # Group 1: Performance metrics (accuracy based)
        performance_metrics = {
            "Grouping Accuracy": metrics_dict["grouping_accuracy"],
            "Parsing Accuracy": metrics_dict["parsing_accuracy"],
            "F1 Grouping Accuracy": metrics_dict["f1_grouping_accuracy"],
            "F1 Template Accuracy": metrics_dict["f1_template_accuracy"]
        }
        
        # Group 2: Granularity metrics (distance based)
        granularity_metrics = {
            "Grouping Granularity Distance": metrics_dict["grouping_granularity_distance"],
            "Parsing Granularity Distance": metrics_dict["parsing_granularity_distance"]
        }
        
        # Group 3: Statistical metrics and metadata
        stats = {
            "Total Logs": metrics_dict["total_logs"],
            "Unique Templates": metrics_dict["unique_templates"],
            "Average Inference Time": f"{metrics_dict['avg_inference_time_ms']:.2f}ms",
            "Model": metrics_dict["model_name"],
            "Total API Calls": metrics_dict.get("total_api_calls", 0),
            "API Calls per Log": f"{metrics_dict.get('total_api_calls', 0) / metrics_dict['total_logs']:.2f}",
            "Cache Hit Rate": f"{metrics_dict.get('cache_hit_rate', 0.0):.1%}"
        }
        
        # Display performance metrics with color coding
        console.print("\n[bold magenta]🎯 Performance Metrics[/bold magenta]")
        for name, value in performance_metrics.items():
            # Color code based on performance thresholds
            if value >= 0.9:
                value_color = "green"
            elif value >= 0.8:
                value_color = "yellow"
            else:
                value_color = "red"
            console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value:.4f}[/bold {value_color}]")
        
        # Display granularity metrics with color coding
        console.print("\n[bold magenta]📏 Granularity Metrics[/bold magenta]")
        for name, value in granularity_metrics.items():
            # Color code based on granularity thresholds
            if value <= 0.1:
                value_color = "green"
            elif value <= 0.2:
                value_color = "yellow"
            else:
                value_color = "red"
            console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value:.4f}[/bold {value_color}]")
        
        # Display statistics with color coding for specific metrics
        console.print("\n[bold magenta]📈 Statistics[/bold magenta]")
        for name, value in stats.items():
            # Color code API calls per log
            if name == "API Calls per Log":
                if float(value.split()[0]) <= 1.0:
                    value_color = "green"
                elif float(value.split()[0]) <= 2.0:
                    value_color = "yellow"
                else:
                    value_color = "red"
                console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value}[/bold {value_color}]")
            # Color code cache hit rate
            elif name == "Cache Hit Rate":
                hit_rate = float(value.strip('%')) / 100
                if hit_rate >= 0.8:
                    value_color = "green"
                elif hit_rate >= 0.6:
                    value_color = "yellow"
                else:
                    value_color = "red"
                console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value}[/bold {value_color}]")
            else:
                console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold white]{value}[/bold white]")
        
        # Display summary footer with overall performance
        console.print("\n[yellow]" + "=" * 50 + "[/yellow]")
        avg_performance = sum(performance_metrics.values()) / len(performance_metrics)
        if avg_performance >= 0.9:
            summary_color = "green"
            emoji = "🌟"
        elif avg_performance >= 0.8:
            summary_color = "yellow"
            emoji = "⭐"
        else:
            summary_color = "red"
            emoji = "⚠️"
        
        console.print(f"\n[bold {summary_color}]{emoji} Overall Performance: {avg_performance:.4f} {emoji}[/bold {summary_color}]")

def run_ui():
    """Launch the Streamlit-based evaluation UI on port 8502."""
    ui_script = Path("src/eval/ui.py")
    if not ui_script.exists():
        console.print("[red]UI script not found at src/eval/ui.py[/red]")
        return False
    
    try:
        subprocess.run(["streamlit", "run", str(ui_script), "--server.port=8502"], check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Failed to start UI: {str(e)}[/red]")
        return False
    except FileNotFoundError:
        console.print("[red]Streamlit command not found. Please ensure streamlit is installed.[/red]")
        return False

def main():
    """Main entry point for the evaluation script.
    
    Handles command line argument parsing and orchestrates the evaluation process.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Run log parser evaluation locally")
    parser.add_argument("--system", default="Apache", choices=["Apache", "Hadoop", "HDFS", "Linux", "OpenStack", "Spark"],
                      help="System to evaluate")
    parser.add_argument("--dataset-type", default="loghub_2k", choices=["loghub_2k", "loghub_all"],
                      help="Dataset type to use")
    parser.add_argument("--ollama-port", default="11434", help="Ollama API port")
    parser.add_argument("--ui", action="store_true", help="Launch the evaluation UI")
    
    args = parser.parse_args()
    
    # Display header
    console.print("\n[bold]Log Parser Evaluation[/bold]")
    console.print("=" * 50)
    
    # Verify all requirements are met
    if not all([
        check_dependencies(),
        check_ollama(),
        check_datasets()
    ]):
        return
    
    # Ensure directory structure exists
    create_directories()
    
    # Run either UI or evaluation based on arguments
    if args.ui:
        console.print("\n[bold]Launching evaluation UI...[/bold]")
        run_ui()
    else:
        console.print("\n[bold]Starting evaluation...[/bold]")
        run_evaluation(args)

if __name__ == "__main__":
    main() 