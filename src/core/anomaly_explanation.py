"""Module for generating LLM-based explanations for log anomalies."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class AnomalyContext:
    """Context information for an anomalous log entry."""
    log_content: str
    cluster_label: int
    is_embedding_anomaly: bool
    is_numeric_anomaly: bool
    component: Optional[str] = None
    level: Optional[str] = None
    numeric_deviations: Optional[Dict[str, float]] = None

class AnomalyExplainer:
    """Generates explanations for anomalous log entries using LLM."""
    
    def __init__(
        self,
        ollama_client: Any,
        max_log_length: int = 200,
        batch_size: int = 10,
        max_retries: int = 3
    ):
        """Initialize the explainer.
        
        Args:
            ollama_client: Ollama client for LLM queries
            max_log_length: Maximum length of log content to include in prompt
            batch_size: Number of anomalies to explain in one batch
            max_retries: Maximum number of retries for failed LLM calls
        """
        self.ollama_client = ollama_client
        self.max_log_length = max_log_length
        self.batch_size = batch_size
        self.max_retries = max_retries
        
    def _build_prompt(self, context: AnomalyContext) -> str:
        """Build prompt for the LLM.
        
        Args:
            context: Anomaly context information
            
        Returns:
            Formatted prompt string
        """
        # Truncate log content if needed
        log_content = context.log_content[:self.max_log_length]
        if len(context.log_content) > self.max_log_length:
            log_content += "..."
            
        # Build reason string
        reasons = []
        if context.is_embedding_anomaly:
            if context.cluster_label == -1:
                reasons.append("Log pattern is an outlier (doesn't match any known clusters)")
            else:
                reasons.append(f"Log belongs to small/unusual cluster (cluster {context.cluster_label})")
                
        if context.is_numeric_anomaly and context.numeric_deviations:
            for field, deviation in context.numeric_deviations.items():
                reasons.append(f"Field '{field}' deviates by {deviation:.1f} standard deviations")
                
        reason_text = "; ".join(reasons)
        
        # Construct the prompt
        prompt = f"""You are an AI assistant that explains why a log entry is suspicious.

Log Entry: {log_content}

Technical Reasons:
- {reason_text}
- Component: {context.component if context.component else 'Unknown'}
- Log Level: {context.level if context.level else 'Unknown'}

Please provide a clear, concise explanation (1-2 sentences) of why this log entry is anomalous.
Focus on the practical implications and potential issues it might indicate.
"""
        return prompt
    
    def _get_explanation(self, context: AnomalyContext) -> str:
        """Get explanation for a single anomaly.
        
        Args:
            context: Anomaly context information
            
        Returns:
            Generated explanation text
        """
        prompt = self._build_prompt(context)
        
        for attempt in range(self.max_retries):
            try:
                response = self.ollama_client.generate(
                    prompt=prompt,
                    max_tokens=100,  # Short response
                    temperature=0.3   # More focused/consistent
                )
                return response.strip()
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to generate explanation after {self.max_retries} attempts: {str(e)}")
                    return "Failed to generate explanation"
                logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                
    def explain_anomalies(self, anomalies_df: pd.DataFrame) -> pd.Series:
        """Generate explanations for anomalous logs.
        
        Args:
            anomalies_df: DataFrame containing anomalous logs
            
        Returns:
            Series containing explanations for each anomaly
        """
        explanations = []
        total_anomalies = len(anomalies_df)
        
        logger.info(f"Generating explanations for {total_anomalies} anomalies")
        
        # Process in batches
        for start_idx in range(0, total_anomalies, self.batch_size):
            batch_df = anomalies_df.iloc[start_idx:start_idx + self.batch_size]
            
            batch_explanations = []
            for _, row in batch_df.iterrows():
                # Extract numeric deviations if present
                numeric_deviations = None
                if row.get('numeric_fields') and row.get('numeric_deviations'):
                    numeric_deviations = dict(zip(
                        row['numeric_fields'],
                        row['numeric_deviations']
                    ))
                
                # Create context
                context = AnomalyContext(
                    log_content=row['message'],
                    cluster_label=row['cluster_label'],
                    is_embedding_anomaly=row['is_embedding_anomaly'],
                    is_numeric_anomaly=row['is_numeric_anomaly'],
                    component=row.get('component'),
                    level=row.get('level'),
                    numeric_deviations=numeric_deviations
                )
                
                # Get explanation
                explanation = self._get_explanation(context)
                batch_explanations.append(explanation)
                
            explanations.extend(batch_explanations)
            logger.info(f"Processed {min(start_idx + self.batch_size, total_anomalies)}/{total_anomalies} anomalies")
            
        return pd.Series(explanations, index=anomalies_df.index) 