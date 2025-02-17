#!/usr/bin/env python3
"""Standalone script for log anomaly detection and analysis."""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.pipeline import LogProcessingPipeline
from src.core.anomaly_incidents import IncidentAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
console = Console()

def setup_detector(args: argparse.Namespace) -> IncidentAnomalyDetector:
    """Initialize the pipeline and detector with given parameters.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configured anomaly detector
    """
    pipeline = LogProcessingPipeline(
        db_url=args.db_url,
        ollama_base_url=args.ollama_url,
        model_name=args.model,
        batch_size=100
    )
    
    detector = IncidentAnomalyDetector(
        pipeline=pipeline,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    return detector

def analyze_historical_trends(
    detector: IncidentAnomalyDetector,
    table_name: str,
    current_window_hours: int,
    progress: Progress
) -> Optional[Dict[str, Any]]:
    """Analyze historical trends for comparison.
    
    Args:
        detector: Anomaly detector instance
        table_name: Name of the logs table
        current_window_hours: Current analysis window in hours
        progress: Rich progress bar
        
    Returns:
        Dictionary containing historical metrics
    """
    end_time = datetime.now()
    windows = []
    window_end = end_time
    
    total_windows = 7 * 24 // current_window_hours
    task = progress.add_task("Analyzing historical windows...", total=total_windows)
    
    for _ in range(total_windows):
        window_start = window_end - timedelta(hours=current_window_hours)
        
        anomalies_df = detector.detect_anomalies(
            table_name=table_name,
            hours=current_window_hours,
            additional_filters={'time_range': (window_start, window_end)}
        )
        
        if not anomalies_df.empty:
            windows.append({
                'start_time': window_start,
                'end_time': window_end,
                'total_anomalies': len(anomalies_df),
                'embedding_anomalies': anomalies_df['is_embedding_anomaly'].sum(),
                'numeric_anomalies': anomalies_df['is_numeric_anomaly'].sum(),
                'unique_components': anomalies_df['component'].nunique(),
                'error_ratio': len(anomalies_df[anomalies_df['level'].isin(['ERROR', 'CRITICAL'])]) / len(anomalies_df)
            })
            
        window_end = window_start
        progress.update(task, advance=1)
    
    if not windows:
        return None
        
    history_df = pd.DataFrame(windows)
    
    return {
        'mean_anomalies': history_df['total_anomalies'].mean(),
        'std_anomalies': history_df['total_anomalies'].std(),
        'p95_anomalies': history_df['total_anomalies'].quantile(0.95),
        'mean_error_ratio': history_df['error_ratio'].mean(),
        'history_df': history_df
    }

def save_visualizations(
    anomalies_df: pd.DataFrame,
    historical_metrics: Optional[Dict[str, Any]],
    output_dir: str
):
    """Save analysis visualizations to files.
    
    Args:
        anomalies_df: Current anomalies DataFrame
        historical_metrics: Historical metrics dictionary
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Timeline visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anomalies_df['timestamp'],
        y=anomalies_df['cluster_label'],
        mode='markers',
        marker=dict(
            color=anomalies_df['is_embedding_anomaly'].map({True: 'red', False: 'blue'}),
            symbol=anomalies_df['is_numeric_anomaly'].map({True: 'star', False: 'circle'})
        ),
        text=anomalies_df['message'],
        name='Anomalies'
    ))
    fig.update_layout(title='Anomaly Timeline')
    fig.write_html(f"{output_dir}/anomaly_timeline_{timestamp}.html")
    
    if historical_metrics:
        # Historical comparison
        fig = make_subplots(rows=2, cols=1)
        history_df = historical_metrics['history_df']
        
        fig.add_trace(
            go.Scatter(
                x=history_df['start_time'],
                y=history_df['total_anomalies'],
                name='Historical Anomalies'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=history_df['start_time'],
                y=history_df['error_ratio'],
                name='Error Ratio'
            ),
            row=2, col=1
        )
        
        fig.update_layout(title='Historical Trends')
        fig.write_html(f"{output_dir}/historical_trends_{timestamp}.html")

def display_summary(
    anomalies_df: pd.DataFrame,
    historical_metrics: Optional[Dict[str, Any]]
):
    """Display analysis summary in the console.
    
    Args:
        anomalies_df: Current anomalies DataFrame
        historical_metrics: Historical metrics dictionary
    """
    # Current anomalies table
    table = Table(title="Anomaly Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Anomalies", str(len(anomalies_df)))
    table.add_row("Embedding Anomalies", str(anomalies_df['is_embedding_anomaly'].sum()))
    table.add_row("Numeric Anomalies", str(anomalies_df['is_numeric_anomaly'].sum()))
    
    if not anomalies_df.empty:
        error_ratio = (
            len(anomalies_df[anomalies_df['level'].isin(['ERROR', 'CRITICAL'])]) / 
            len(anomalies_df) * 100
        )
        table.add_row("Error Ratio", f"{error_ratio:.1f}%")
    
    console.print(table)
    
    # Historical comparison if available
    if historical_metrics:
        hist_table = Table(title="Historical Comparison")
        hist_table.add_column("Metric", style="cyan")
        hist_table.add_column("Value", style="magenta")
        
        current_total = len(anomalies_df)
        z_score = (
            (current_total - historical_metrics['mean_anomalies']) / 
            historical_metrics['std_anomalies']
        )
        
        hist_table.add_row("Historical Mean", f"{historical_metrics['mean_anomalies']:.1f}")
        hist_table.add_row("Current vs Mean", f"{z_score:+.2f}σ")
        hist_table.add_row("95th Percentile", f"{historical_metrics['p95_anomalies']:.1f}")
        
        console.print(hist_table)

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Analyze logs for anomalies with historical comparison'
    )
    
    parser.add_argument(
        '--hours',
        type=int,
        default=4,
        help='Number of hours to analyze (default: 4)'
    )
    parser.add_argument(
        '--table',
        type=str,
        default='logs',
        help='Name of the logs table (default: logs)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DB_URL', 'postgresql://localhost:5432/logs'),
        help='Database URL'
    )
    parser.add_argument(
        '--ollama-url',
        type=str,
        default='http://localhost:11434',
        help='Ollama service URL'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='mistral',
        help='Model name for log parsing'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=0.3,
        help='DBSCAN epsilon parameter (default: 0.3)'
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=3,
        help='DBSCAN min_samples parameter (default: 3)'
    )
    parser.add_argument(
        '--level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Filter by log level'
    )
    parser.add_argument(
        '--component',
        type=str,
        help='Filter by component name'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='anomaly_reports',
        help='Directory for saving visualizations and reports'
    )
    parser.add_argument(
        '--no-history',
        action='store_true',
        help='Skip historical analysis'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = setup_detector(args)
        
        # Build filters
        filters = {}
        if args.level:
            filters['level'] = args.level
        if args.component:
            filters['component'] = args.component
            
        with Progress() as progress:
            # Detect current anomalies
            task = progress.add_task("Detecting anomalies...", total=1)
            anomalies_df = detector.detect_anomalies(
                table_name=args.table,
                hours=args.hours,
                additional_filters=filters
            )
            progress.update(task, completed=1)
            
            # Analyze historical trends
            historical_metrics = None
            if not args.no_history and not anomalies_df.empty:
                historical_metrics = analyze_historical_trends(
                    detector,
                    args.table,
                    args.hours,
                    progress
                )
        
        # Display results
        if anomalies_df.empty:
            console.print("[yellow]No anomalies detected in the specified time window[/yellow]")
        else:
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            
            # Save anomalies CSV
            csv_path = f"{output_dir}/anomalies_{timestamp}.csv"
            anomalies_df.to_csv(csv_path, index=False)
            console.print(f"[green]Saved anomalies to {csv_path}[/green]")
            
            # Generate and save visualizations
            save_visualizations(anomalies_df, historical_metrics, output_dir)
            console.print(f"[green]Saved visualizations to {output_dir}[/green]")
            
            # Display summary
            display_summary(anomalies_df, historical_metrics)
            
    except Exception as e:
        console.print(f"[red]Error analyzing logs: {str(e)}[/red]")
        raise
        
    finally:
        detector.pipeline.close()

if __name__ == '__main__':
    main() 