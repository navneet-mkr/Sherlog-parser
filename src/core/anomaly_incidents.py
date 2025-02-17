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

logger = logging.getLogger(__name__)

class IncidentAnomalyDetector:
    """Detector for anomalous logs during incidents using local clustering."""
    
    def __init__(
        self,
        pipeline: LogProcessingPipeline,
        eps: float = 0.3,  # Smaller eps for higher recall
        min_samples: int = 3,  # Smaller min_samples for higher recall
        numeric_std_threshold: float = 2.5
    ):
        """Initialize the anomaly detector.
        
        Args:
            pipeline: Existing log processing pipeline
            eps: DBSCAN epsilon parameter (distance threshold)
            min_samples: DBSCAN min samples for core points
            numeric_std_threshold: Standard deviations for numeric outliers
        """
        self.pipeline = pipeline
        self.eps = eps
        self.min_samples = min_samples
        self.numeric_std_threshold = numeric_std_threshold
        
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