#!/usr/bin/env python3

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import streamlit.cli as stcli
from dataclasses import asdict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

def check_dependencies():
    """Check if all required dependencies are installed."""
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
    """Check if Ollama is running and has the required model."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise ConnectionError("Ollama is not running")
        
        tags = response.json()
        if not any(tag.get('name') == 'mistral' for tag in tags.get('models', [])):
            console.print("[yellow]Mistral model not found. Pulling...[/yellow]")
            requests.post("http://localhost:11434/api/pull", json={"name": "mistral"})
        return True
    except Exception as e:
        console.print(f"[red]Error connecting to Ollama: {str(e)}[/red]")
        console.print("\nPlease ensure Ollama is running:")
        console.print("1. Install Ollama from https://ollama.ai")
        console.print("2. Start Ollama service")
        console.print("3. Pull the Mistral model: ollama pull mistral")
        return False

def check_datasets():
    """Check if evaluation datasets are available."""
    dataset_path = Path("./data/eval_datasets")
    if not dataset_path.exists():
        console.print("[yellow]Evaluation datasets not found[/yellow]")
        console.print("\nPlease download the datasets:")
        console.print("1. Create data directory: mkdir -p data/eval_datasets")
        console.print("2. Download and extract datasets into data/eval_datasets")
        return False
    return True

def create_directories():
    """Create necessary directories."""
    directories = [
        "data/eval_datasets",
        "output/eval",
        "cache/eval",
        "logs"
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def run_evaluation(args):
    """Run the evaluation process."""
    from src.core.eval import Evaluator
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold yellow][[/bold yellow][progress.description]{task.description}[bold yellow]][/bold yellow]"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[bold yellow]•[/bold yellow]"),
        TextColumn("[bold blue]{task.fields[status]}"),
        transient=True,
    ) as progress:
        # Main progress task
        main_task = progress.add_task(
            "[bold magenta]Evaluation Progress[/bold magenta]",
            total=100,
            status="Initializing..."
        )
        
        # Update initialization progress
        progress.update(main_task, status="[bold green]Loading evaluator...[/bold green]")
        evaluator = Evaluator(
            base_dir="./data/eval_datasets",
            dataset_type=args.dataset_type,
            system=args.system,
            llm_model="mistral",
            llm_api_base=f"http://localhost:{args.ollama_port}",
            output_dir="./output/eval",
            cache_dir="./cache/eval"
        )
        progress.update(main_task, completed=10, status="[bold green]Evaluator loaded[/bold green]")
        
        # Start evaluation
        progress.update(main_task, completed=15, status="[bold yellow]Starting evaluation...[/bold yellow]")
        metrics = evaluator.evaluate()
        
        # Update progress for results saving
        progress.update(main_task, completed=90, status="[bold yellow]Saving results...[/bold yellow]")
        
        # Save results
        dataset_name = f"{args.system}_{args.dataset_type}"
        output_dir = Path("./output/eval")
        with open(output_dir / f"{dataset_name}_metrics.json", 'w') as f:
            json.dump(asdict(metrics), f, indent=2)
        
        # Complete progress
        progress.update(main_task, completed=100, status="[bold green]Complete![/bold green]")
        
        # Display results header with rainbow effect
        console.print("\n[bold green]✨ Evaluation complete! ✨[/bold green]")
        console.print("\n[bold magenta]📊 Evaluation Results[/bold magenta]")
        console.print("[yellow]" + "=" * 50 + "[/yellow]")
        
        metrics_dict = asdict(metrics)
        # Group metrics by type
        performance_metrics = {
            "Grouping Accuracy": metrics_dict["grouping_accuracy"],
            "Parsing Accuracy": metrics_dict["parsing_accuracy"],
            "F1 Grouping Accuracy": metrics_dict["f1_grouping_accuracy"],
            "F1 Template Accuracy": metrics_dict["f1_template_accuracy"]
        }
        
        granularity_metrics = {
            "Grouping Granularity Distance": metrics_dict["grouping_granularity_distance"],
            "Parsing Granularity Distance": metrics_dict["parsing_granularity_distance"]
        }
        
        stats = {
            "Total Logs": metrics_dict["total_logs"],
            "Unique Templates": metrics_dict["unique_templates"],
            "Average Inference Time": f"{metrics_dict['avg_inference_time_ms']:.2f}ms",
            "Model": metrics_dict["model_name"]
        }
        
        # Display grouped metrics with enhanced colors
        console.print("\n[bold magenta]🎯 Performance Metrics[/bold magenta]")
        for name, value in performance_metrics.items():
            if value >= 0.9:
                value_color = "green"
            elif value >= 0.8:
                value_color = "yellow"
            else:
                value_color = "red"
            console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value:.4f}[/bold {value_color}]")
        
        console.print("\n[bold magenta]📏 Granularity Metrics[/bold magenta]")
        for name, value in granularity_metrics.items():
            if value <= 0.1:
                value_color = "green"
            elif value <= 0.2:
                value_color = "yellow"
            else:
                value_color = "red"
            console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold {value_color}]{value:.4f}[/bold {value_color}]")
        
        console.print("\n[bold magenta]📈 Statistics[/bold magenta]")
        for name, value in stats.items():
            console.print(f"[cyan]{name}[/cyan][yellow]{'.' * (40 - len(name))}[/yellow][bold white]{value}[/bold white]")
        
        # Add a summary footer
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
    """Run the Streamlit UI."""
    ui_script = Path("src/eval/ui.py")
    if not ui_script.exists():
        console.print("[red]UI script not found at src/eval/ui.py[/red]")
        return False
    
    sys.argv = ["streamlit", "run", str(ui_script), "--server.port=8502"]
    stcli.main()

def main():
    parser = argparse.ArgumentParser(description="Run log parser evaluation locally")
    parser.add_argument("--system", default="Apache", choices=["Apache", "Hadoop", "HDFS", "Linux", "OpenStack", "Spark"],
                      help="System to evaluate")
    parser.add_argument("--dataset-type", default="loghub_2k", choices=["loghub_2k", "loghub_all"],
                      help="Dataset type to use")
    parser.add_argument("--ollama-port", default="11434", help="Ollama API port")
    parser.add_argument("--ui", action="store_true", help="Launch the evaluation UI")
    
    args = parser.parse_args()
    
    # Print header
    console.print("\n[bold]Log Parser Evaluation[/bold]")
    console.print("=" * 50)
    
    # Check requirements
    if not all([
        check_dependencies(),
        check_ollama(),
        check_datasets()
    ]):
        return
    
    # Create directories
    create_directories()
    
    if args.ui:
        console.print("\n[bold]Launching evaluation UI...[/bold]")
        run_ui()
    else:
        console.print("\n[bold]Starting evaluation...[/bold]")
        run_evaluation(args)

if __name__ == "__main__":
    main() 