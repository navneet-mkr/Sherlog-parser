"""Module providing LogAI-based tools for the agent."""

import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.core.logai_tools import LogAIAnalyzer

logger = logging.getLogger(__name__)
console = Console()

class LogAITools:
    """Class providing LogAI analysis capabilities as agent tools."""
    
    def __init__(self, log_file: str):
        """Initialize LogAI tools.
        
        Args:
            log_file: Path to the log file to analyze
        """
        self.analyzer = LogAIAnalyzer(log_file)
        
    def analyze_time_patterns(
        self,
        time_window: str = "15min",
        algorithm: str = "dbl",
        group_by: Optional[List[str]] = None,
        explanation: bool = True
    ) -> Dict[str, Any]:
        """Analyze temporal patterns in logs using LogAI.
        
        Args:
            time_window: Time window for aggregation
            algorithm: Algorithm to use ('dbl', 'isolation_forest')
            group_by: Fields to group by
            explanation: Whether to include explanation
            
        Returns:
            Dictionary with analysis results and metadata
        """
        # Detect anomalies
        anomalies, metadata = self.analyzer.detect_timeseries_anomalies(
            time_window=time_window,
            algorithm=algorithm,
            group_by=group_by
        )
        
        # Create visualization
        if not anomalies.empty:
            viz_path = self.analyzer.visualize_anomalies(anomalies)
        else:
            viz_path = None
            
        # Prepare response
        response = {
            'success': True,
            'anomalies': anomalies.to_dict('records') if not anomalies.empty else [],
            'metadata': metadata,
            'visualization': viz_path
        }
        
        if explanation:
            response['explanation'] = self._explain_time_patterns(anomalies, metadata)
            
        return response
        
    def cluster_similar_logs(
        self,
        n_clusters: int = 10,
        use_attributes: bool = True,
        explanation: bool = True
    ) -> Dict[str, Any]:
        """Cluster similar log messages using LogAI.
        
        Args:
            n_clusters: Number of clusters to form
            use_attributes: Whether to use log attributes
            explanation: Whether to include explanation
            
        Returns:
            Dictionary with clustering results and metadata
        """
        # Perform clustering
        clustered_df, stats = self.analyzer.cluster_logs(
            n_clusters=n_clusters,
            use_attributes=use_attributes
        )
        
        # Create visualization
        viz_path = self.analyzer.visualize_clusters(clustered_df)
        
        # Prepare response
        response = {
            'success': True,
            'clusters': clustered_df.to_dict('records'),
            'statistics': stats,
            'visualization': viz_path
        }
        
        if explanation:
            response['explanation'] = self._explain_clusters(clustered_df, stats)
            
        return response
        
    def find_semantic_anomalies(
        self,
        algorithm: str = "isolation_forest",
        train_size: float = 0.7,
        explanation: bool = True
    ) -> Dict[str, Any]:
        """Find semantically anomalous logs using LogAI.
        
        Args:
            algorithm: Algorithm to use
            train_size: Fraction of data for training
            explanation: Whether to include explanation
            
        Returns:
            Dictionary with anomaly results and metadata
        """
        # Detect anomalies
        anomalies_df, metadata = self.analyzer.detect_semantic_anomalies(
            algorithm=algorithm,
            train_size=train_size
        )
        
        # Create visualization
        if not anomalies_df.empty:
            viz_path = self.analyzer.visualize_anomalies(anomalies_df)
        else:
            viz_path = None
            
        # Prepare response
        response = {
            'success': True,
            'anomalies': anomalies_df.to_dict('records') if not anomalies_df.empty else [],
            'metadata': metadata,
            'visualization': viz_path
        }
        
        if explanation:
            response['explanation'] = self._explain_semantic_anomalies(anomalies_df, metadata)
            
        return response
        
    def display_analysis_summary(self, results: Dict[str, Any]):
        """Display analysis results in a user-friendly format.
        
        Args:
            results: Analysis results dictionary
        """
        if not results.get('success'):
            console.print("[red]Analysis failed[/red]")
            return
            
        # Create summary table
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        if 'metadata' in results:
            for key, value in results['metadata'].items():
                table.add_row(key.replace('_', ' ').title(), str(value))
                
        if 'statistics' in results:
            for key, value in results['statistics'].items():
                if key == 'cluster_sizes':
                    table.add_row(
                        "Cluster Distribution",
                        ", ".join(f"C{k}: {v}" for k, v in value.items())
                    )
                else:
                    table.add_row(key.replace('_', ' ').title(), f"{value:.3f}")
                    
        console.print(table)
        
        # Show explanation if available
        if 'explanation' in results:
            console.print("\n[bold]Analysis Explanation:[/bold]")
            console.print(results['explanation'])
            
        # Show visualization path if available
        if results.get('visualization'):
            console.print(f"\n[green]Visualization saved to: {results['visualization']}[/green]")
            
    def _explain_time_patterns(
        self,
        anomalies: Any,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate explanation for time-series analysis results.
        
        Args:
            anomalies: Anomaly detection results
            metadata: Analysis metadata
            
        Returns:
            Human-readable explanation
        """
        if anomalies.empty:
            return "No significant temporal anomalies were detected in the log patterns."
            
        explanation = [
            f"Analysis detected {metadata['anomaly_windows']} anomalous time windows "
            f"out of {metadata['total_windows']} total windows "
            f"(using {metadata['time_window']} intervals)."
        ]
        
        if 'error_ratio' in metadata:
            explanation.append(
                f"The anomalous windows have an error ratio of {metadata['error_ratio']:.1%}."
            )
            
        return " ".join(explanation)
        
    def _explain_clusters(
        self,
        clustered_df: Any,
        stats: Dict[str, Any]
    ) -> str:
        """Generate explanation for clustering results.
        
        Args:
            clustered_df: Clustering results
            stats: Clustering statistics
            
        Returns:
            Human-readable explanation
        """
        cluster_sizes = stats['cluster_sizes']
        silhouette = stats.get('silhouette_score', 0)
        
        explanation = [
            f"Logs were grouped into {len(cluster_sizes)} distinct clusters. "
            f"The largest cluster contains {max(cluster_sizes.values())} logs, "
            f"while the smallest has {min(cluster_sizes.values())} logs."
        ]
        
        if silhouette > 0:
            quality = "excellent" if silhouette > 0.7 else \
                     "good" if silhouette > 0.5 else \
                     "fair" if silhouette > 0.3 else "poor"
                     
            explanation.append(
                f"The clustering quality is {quality} "
                f"(silhouette score: {silhouette:.2f})."
            )
            
        return " ".join(explanation)
        
    def _explain_semantic_anomalies(
        self,
        anomalies_df: Any,
        metadata: Dict[str, Any]
    ) -> str:
        """Generate explanation for semantic anomaly results.
        
        Args:
            anomalies_df: Anomaly detection results
            metadata: Analysis metadata
            
        Returns:
            Human-readable explanation
        """
        if anomalies_df.empty:
            return "No semantic anomalies were detected in the logs."
            
        explanation = [
            f"Analysis found {metadata['anomaly_count']} semantically anomalous logs "
            f"out of {metadata['total_logs']} total logs "
            f"({metadata['anomaly_rate']:.1%} anomaly rate)."
        ]
        
        # Add component/level distribution if available
        if 'Level' in anomalies_df.columns:
            level_counts = anomalies_df['Level'].value_counts()
            explanation.append(
                f"The anomalies include: " + 
                ", ".join(f"{count} {level.lower()} messages" 
                         for level, count in level_counts.items())
            )
            
        return " ".join(explanation) 