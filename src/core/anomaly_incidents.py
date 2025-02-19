"""Module for real-time log anomaly detection during incidents."""

import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from src.core.pipeline import LogProcessingPipeline
from src.core.timeseries import LogTimeSeriesDB
from src.core.anomaly_explanation import AnomalyExplainer
from src.core.log_prefilter import LogPreFilter, PreFilterConfig

logger = logging.getLogger(__name__)

class IncidentAnomalyDetector:
    """Detector for anomalous logs during incidents using local clustering."""
    
    def __init__(
        self,
        pipeline: LogProcessingPipeline,
        eps: float = 0.3,  # Smaller eps for higher recall
        min_samples: int = 3,  # Smaller min_samples for higher recall
        numeric_std_threshold: float = 2.5,
        explain_anomalies: bool = True,  # Whether to generate explanations
        max_explanations: int = 100,  # Maximum number of anomalies to explain
        enable_prefilter: bool = False,  # Whether to enable pre-filtering
        prefilter_config: Optional[PreFilterConfig] = None  # Pre-filter configuration
    ):
        """Initialize the anomaly detector.
        
        Args:
            pipeline: Existing log processing pipeline
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: DBSCAN min samples for core points
            numeric_std_threshold: Standard deviations for numeric outliers
            explain_anomalies: Whether to generate LLM explanations
            max_explanations: Maximum number of anomalies to explain
            enable_prefilter: Whether to enable pre-filtering
            prefilter_config: Pre-filter configuration
        """
        self.pipeline = pipeline
        self.eps = eps
        self.min_samples = min_samples
        self.numeric_std_threshold = numeric_std_threshold
        self.explain_anomalies = explain_anomalies
        self.max_explanations = max_explanations
        self.enable_prefilter = enable_prefilter
        
        if explain_anomalies:
            self.explainer = AnomalyExplainer(
                ollama_client=pipeline.ollama_analyzer
            )
            
        if enable_prefilter:
            self.prefilter = LogPreFilter(prefilter_config)

    def detect_anomalies(
        self,
        table_name: str,
        hours: int = 4,
        additional_filters: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Detect anomalies in logs from the last N hours.
        
        Args:
            table_name: Name of the logs table
            hours: Number of hours to look back
            additional_filters: Optional dict of additional query filters
            
        Returns:
            DataFrame containing anomalous logs with detection metadata
        """
        # Query recent logs
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get logs using existing pipeline query
        logs_df = self.pipeline.query_logs(
            table_name=table_name,
            time_range=(start_time, end_time),
            limit=100000,  # High limit for incident analysis
            **additional_filters if additional_filters else {}
        )
        
        if logs_df.empty:
            logger.warning(f"No logs found in the last {hours} hours")
            return pd.DataFrame()
            
        # Apply pre-filtering if enabled
        original_size = len(logs_df)
        if self.enable_prefilter:
            logs_df = self.prefilter.filter_logs(logs_df)
            if logs_df.empty:
                logger.warning("Pre-filter removed all logs, using original dataset")
                logs_df = self.pipeline.query_logs(
                    table_name=table_name,
                    time_range=(start_time, end_time),
                    limit=100000,
                    **additional_filters if additional_filters else {}
                )
            elif len(logs_df) < self.min_samples:
                logger.warning(
                    f"Pre-filter reduced logs below min_samples ({len(logs_df)} < {self.min_samples}), "
                    "using original dataset"
                )
                logs_df = self.pipeline.query_logs(
                    table_name=table_name,
                    time_range=(start_time, end_time),
                    limit=100000,
                    **additional_filters if additional_filters else {}
                )
                
        # Get embeddings from existing parsed results
        embeddings = np.array(logs_df['embedding'].tolist())
        
        # Cluster using DBSCAN
        dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric='cosine'
        )
        labels = dbscan.fit_predict(embeddings)
        
        # Add clustering results
        logs_df['cluster_label'] = labels
        logs_df['is_embedding_anomaly'] = (labels == -1)
        
        # Find small clusters (potential group anomalies)
        cluster_counts = logs_df['cluster_label'].value_counts()
        small_clusters = cluster_counts[cluster_counts < self.min_samples].index
        small_clusters = small_clusters[small_clusters != -1]  # Exclude noise points
        logs_df.loc[logs_df['cluster_label'].isin(small_clusters), 'is_embedding_anomaly'] = True
        
        # Check numeric fields if present
        numeric_anomalies = self._detect_numeric_anomalies(logs_df)
        logs_df['is_numeric_anomaly'] = numeric_anomalies
        
        # Combine anomaly flags
        logs_df['is_anomaly'] = (
            logs_df['is_embedding_anomaly'] | 
            logs_df['is_numeric_anomaly']
        )
        
        # Return anomalous logs with metadata
        anomalies_df = logs_df[logs_df['is_anomaly']].copy()
        
        # Add detection metadata
        anomalies_df['detection_time'] = datetime.now()
        anomalies_df['detection_window_hours'] = hours
        
        # Generate explanations if enabled
        if self.explain_anomalies and not anomalies_df.empty:
            # Store numeric deviations for explanation context
            if 'numeric_anomalies' in locals():
                # Create empty columns for numeric info
                anomalies_df['numeric_fields'] = [[] for _ in range(len(anomalies_df))]
                anomalies_df['numeric_deviations'] = [[] for _ in range(len(anomalies_df))]
                
                # Process each numeric anomaly
                for field, anomaly_mask in numeric_anomalies.items():
                    # Get anomalous rows for this field
                    field_anomalies = anomalies_df[anomaly_mask]
                    
                    for idx in field_anomalies.index:
                        row = field_anomalies.loc[idx]
                        cluster = row['cluster_label']
                        
                        # Calculate deviation if in a valid cluster
                        if cluster != -1:
                            cluster_values = logs_df[logs_df['cluster_label'] == cluster][field]
                            if len(cluster_values) >= 2:
                                mean = cluster_values.mean()
                                std = cluster_values.std()
                                if std > 0:
                                    dev = (float(row[field]) - mean) / std
                                    # Append to existing lists
                                    anomalies_df.at[idx, 'numeric_fields'].append(field)
                                    anomalies_df.at[idx, 'numeric_deviations'].append(dev)
            
            # Generate explanations (limit to max_explanations)
            explanation_df = anomalies_df.head(self.max_explanations)
            explanations = self.explainer.explain_anomalies(explanation_df)
            
            # Add explanations to the full DataFrame
            anomalies_df['explanation'] = pd.NA
            anomalies_df.loc[explanations.index, 'explanation'] = explanations
            
            if len(anomalies_df) > self.max_explanations:
                logger.info(
                    f"Generated explanations for first {self.max_explanations} "
                    f"of {len(anomalies_df)} anomalies"
                )
        
        # Sort by timestamp
        anomalies_df = anomalies_df.sort_values('timestamp', ascending=False)
        
        logger.info(
            f"Found {len(anomalies_df)} anomalies in {len(logs_df)} logs "
            f"({len(anomalies_df)/len(logs_df)*100:.1f}% anomaly rate)"
        )
        
        return anomalies_df
    
    def _detect_numeric_anomalies(self, logs_df: pd.DataFrame) -> pd.Series:
        """Detect anomalies in numeric fields within clusters.
        
        Args:
            logs_df: DataFrame containing logs with cluster labels
            
        Returns:
            Boolean series indicating numeric anomalies
        """
        numeric_anomalies = pd.Series(False, index=logs_df.index)
        
        # Get numeric columns from parameters
        numeric_fields = self._extract_numeric_fields(logs_df)
        
        if not numeric_fields:
            return numeric_anomalies
            
        # Check each numeric field within its cluster
        for cluster in logs_df['cluster_label'].unique():
            if cluster == -1:  # Skip noise points
                continue
                
            cluster_mask = logs_df['cluster_label'] == cluster
            cluster_data = logs_df[cluster_mask]
            
            for field in numeric_fields:
                values = cluster_data[field].astype(float)
                if len(values) < 2:  # Need at least 2 points for std
                    continue
                    
                mean = values.mean()
                std = values.std()
                
                if std == 0:  # Skip constant values
                    continue
                    
                # Mark values beyond threshold standard deviations
                outliers = abs(values - mean) > (self.numeric_std_threshold * std)
                numeric_anomalies[cluster_data[outliers].index] = True
                
        return numeric_anomalies
    
    def _extract_numeric_fields(self, logs_df: pd.DataFrame) -> List[str]:
        """Extract numeric fields from log parameters.
        
        Args:
            logs_df: DataFrame containing logs
            
        Returns:
            List of numeric field names
        """
        numeric_fields = []
        
        # Check parameters column for numeric values
        if 'parameters' not in logs_df.columns:
            return numeric_fields
            
        # Sample some non-null parameters to identify numeric fields
        sample_params = logs_df['parameters'].dropna().head(100)
        
        # Collect fields that are consistently numeric
        field_types = {}
        
        for params in sample_params:
            for key, value in params.items():
                try:
                    float(value)
                    field_types[key] = field_types.get(key, 0) + 1
                except (ValueError, TypeError):
                    pass
                    
        # Consider fields numeric if they are numeric in >80% of samples
        threshold = len(sample_params) * 0.8
        numeric_fields = [
            field for field, count in field_types.items()
            if count >= threshold
        ]
        
        return numeric_fields 