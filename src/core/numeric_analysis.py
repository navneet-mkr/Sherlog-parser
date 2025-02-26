"""Module for detecting anomalies in numeric log fields."""

import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

class NumericAnomalyDetector:
    """Detector for anomalies in numeric log fields using statistical methods."""
    
    def __init__(
        self,
        std_threshold: float = 3.0,
        iqr_threshold: float = 1.5,
        min_samples: int = 10,
        use_robust: bool = True
    ):
        """Initialize the numeric anomaly detector.
        
        Args:
            std_threshold: Number of standard deviations for outlier detection
            iqr_threshold: IQR multiplier for outlier detection
            min_samples: Minimum samples needed for analysis
            use_robust: Whether to use robust statistics (median/IQR vs mean/std)
        """
        self.std_threshold = std_threshold
        self.iqr_threshold = iqr_threshold
        self.min_samples = min_samples
        self.use_robust = use_robust
        
    def detect_anomalies(
        self,
        df: pd.DataFrame,
        numeric_fields: Optional[List[str]] = None,
        group_by: Optional[str] = None
    ) -> Tuple[pd.Series, Dict[str, pd.Series]]:
        """Detect anomalies in numeric fields.
        
        Args:
            df: DataFrame containing logs
            numeric_fields: List of numeric column names to analyze
            group_by: Optional column name to group by (e.g. 'cluster_label')
            
        Returns:
            Tuple of (combined anomaly mask, per-field anomaly masks)
        """
        if numeric_fields is None:
            numeric_fields = self._extract_numeric_fields(df)
            
        if not numeric_fields:
            return pd.Series(False, index=df.index), {}
            
        field_anomalies = {}
        combined_mask = pd.Series(False, index=df.index)
        
        # Process each numeric field
        for field in numeric_fields:
            if field not in df.columns:
                continue
                
            values = pd.to_numeric(df[field], errors='coerce')
            if values.isna().all():
                continue
                
            # Detect anomalies for this field
            if group_by and group_by in df.columns:
                # Detect within groups
                anomaly_mask = pd.Series(False, index=df.index)
                for group in df[group_by].unique():
                    if group == -1:  # Skip noise points in clustering
                        continue
                    group_mask = df[group_by] == group
                    group_values = values[group_mask]
                    
                    if len(group_values) >= self.min_samples:
                        group_anomalies = self._detect_field_anomalies(group_values)
                        anomaly_mask[group_mask] = group_anomalies
            else:
                # Detect across all values
                if len(values) >= self.min_samples:
                    anomaly_mask = self._detect_field_anomalies(values)
                else:
                    anomaly_mask = pd.Series(False, index=df.index)
            
            field_anomalies[field] = anomaly_mask
            combined_mask |= anomaly_mask
            
        return combined_mask, field_anomalies
        
    def _detect_field_anomalies(self, values: pd.Series) -> pd.Series:
        """Detect anomalies in a single numeric field.
        
        Args:
            values: Series of numeric values
            
        Returns:
            Boolean series marking anomalies
        """
        if self.use_robust:
            # Use robust statistics (median/IQR)
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            
            if iqr == 0:  # Handle zero IQR case
                return pd.Series(False, index=values.index)
                
            lower = q1 - (self.iqr_threshold * iqr)
            upper = q3 + (self.iqr_threshold * iqr)
            
            return (values < lower) | (values > upper)
        else:
            # Use classical statistics (mean/std)
            mean = values.mean()
            std = values.std()
            
            if std == 0:  # Handle zero std case
                return pd.Series(False, index=values.index)
                
            z_scores = np.abs((values - mean) / std)
            return pd.Series(z_scores > self.std_threshold, index=values.index)
            
    def _extract_numeric_fields(self, df: pd.DataFrame) -> List[str]:
        """Extract names of numeric columns from DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of numeric column names
        """
        numeric_fields = []
        
        for col in df.columns:
            # Skip known non-numeric columns
            if col in ['message', 'component', 'level', 'timestamp']:
                continue
                
            # Try converting to numeric
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_fields.append(col)
            except (ValueError, TypeError):
                continue
                
        return numeric_fields 