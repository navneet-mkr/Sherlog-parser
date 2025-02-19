"""Module for pre-filtering log data before expensive operations."""

import logging
from typing import List, Dict, Any, Optional, Set, Union
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PreFilterConfig:
    """Configuration for log pre-filtering."""
    # Priority levels to always keep
    priority_levels: Set[str] = field(
        default_factory=lambda: {'ERROR', 'CRITICAL'}
    )
    
    # Sampling ratios for different log levels
    level_sample_ratios: Dict[str, float] = field(
        default_factory=lambda: {
            'INFO': 0.1,
            'DEBUG': 0.05,
            'WARNING': 0.5
        }
    )
    
    # Default sampling ratio for unspecified levels
    default_sample_ratio: float = 0.1
    
    # Minimum number of logs to keep per level
    min_logs_per_level: int = 10
    
    # Maximum number of duplicate messages to keep
    max_duplicates: Optional[int] = 5
    
    def __post_init__(self):
        """Initialize default values."""
        if self.level_sample_ratios is None:
            self.level_sample_ratios = {
                'INFO': 0.1,
                'DEBUG': 0.05,
                'WARNING': 0.5
            }

class LogPreFilter:
    """Pre-filters logs before expensive operations like embedding and clustering."""
    
    def __init__(self, config: Optional[PreFilterConfig] = None):
        """Initialize the pre-filter.
        
        Args:
            config: Pre-filter configuration
        """
        self.config = config or PreFilterConfig()
        
    def _sample_by_level(
        self,
        logs_df: pd.DataFrame,
        level: str,
        ratio: float
    ) -> pd.DataFrame:
        """Sample logs of a specific level.
        
        Args:
            logs_df: DataFrame containing logs
            level: Log level to sample
            ratio: Sampling ratio (0.0 to 1.0)
            
        Returns:
            Sampled DataFrame
        """
        level_df = logs_df[logs_df['level'] == level]
        
        if level_df.empty:
            return level_df
            
        # Calculate sample size
        sample_size = max(
            self.config.min_logs_per_level,
            int(len(level_df) * ratio)
        )
        
        # Ensure we don't try to sample more than we have
        sample_size = min(sample_size, len(level_df))
        
        return level_df.sample(n=sample_size, random_state=42)
        
    def _deduplicate_messages(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Reduce duplicate messages while preserving time distribution.
        
        Args:
            logs_df: DataFrame containing logs
            
        Returns:
            Deduplicated DataFrame
        """
        if self.config.max_duplicates is None or logs_df.empty:
            return logs_df
            
        # Group by message and get counts
        message_counts = logs_df.groupby('message').size()
        
        # Identify messages that exceed the duplicate threshold
        threshold = self.config.max_duplicates
        excessive_duplicates = message_counts[np.array(message_counts) > threshold].index
        
        if excessive_duplicates.empty:
            return logs_df
            
        # Process each message that has too many duplicates
        filtered_dfs = []
        for message in excessive_duplicates:
            message_df = logs_df[logs_df['message'] == message]
            
            # Keep first, last, and sampled middle entries
            middle_size = self.config.max_duplicates - 2
            if middle_size > 0:
                middle_df = message_df.iloc[1:-1].sample(
                    n=min(middle_size, len(message_df) - 2),
                    random_state=42
                )
                kept_df = pd.concat([
                    message_df.iloc[[0]],  # First occurrence
                    middle_df,             # Sampled middle
                    message_df.iloc[[-1]]  # Last occurrence
                ])
            else:
                # If max_duplicates <= 2, just keep first and last
                kept_df = pd.concat([
                    message_df.iloc[[0]],
                    message_df.iloc[[-1]]
                ])
                
            filtered_dfs.append(kept_df)
            
        # Combine with messages that weren't over the threshold
        non_duplicate_df = logs_df[~logs_df['message'].isin(excessive_duplicates)]
        filtered_dfs.append(non_duplicate_df)
        
        return pd.concat(filtered_dfs, ignore_index=True)
    
    def filter_logs(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Apply pre-filtering to reduce log volume.
        
        Args:
            logs_df: DataFrame containing logs
            
        Returns:
            Filtered DataFrame
        """
        if logs_df.empty:
            return logs_df
            
        # Start with priority logs
        priority_df = logs_df[logs_df['level'].isin(self.config.priority_levels)]
        
        # Sample other levels
        sampled_dfs = []
        for level in logs_df['level'].unique():
            if level not in self.config.priority_levels:
                ratio = self.config.level_sample_ratios.get(
                    level,
                    self.config.default_sample_ratio
                )
                sampled_df = self._sample_by_level(logs_df, level, ratio)
                sampled_dfs.append(sampled_df)
                
        # Combine all
        filtered_df = pd.concat([priority_df] + sampled_dfs, ignore_index=True)
        
        # Deduplicate if configured
        if self.config.max_duplicates is not None:
            filtered_df = self._deduplicate_messages(filtered_df)
            
        # Sort by timestamp to maintain temporal order
        filtered_df = filtered_df.sort_values('timestamp')
        
        # Log reduction stats
        reduction = (1 - len(filtered_df) / len(logs_df)) * 100
        logger.info(
            f"Pre-filter reduced log volume by {reduction:.1f}% "
            f"({len(filtered_df)} / {len(logs_df)} logs kept)"
        )
        
        return filtered_df 