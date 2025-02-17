#!/usr/bin/env python3
"""CLI tool for analyzing log anomalies during incidents."""

import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from src.core.pipeline import LogProcessingPipeline
from src.core.anomaly_incidents import IncidentAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze logs for anomalies during incidents'
    )
    parser.add_argument(
        '--hours',
        type=int,
        default=4,
        help='Number of hours to look back (default: 4)'
    )
    parser.add_argument(
        '--table',
        type=str,
        required=True,
        help='Name of the logs table to analyze'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for anomalies (optional)'
    )
    parser.add_argument(
        '--db-url',
        type=str,
        default=os.getenv('DB_URL', 'postgresql://localhost:5432/logs'),
        help='Database URL (default: from DB_URL env var or localhost)'
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
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Filter by log level (optional)'
    )
    parser.add_argument(
        '--component',
        type=str,
        help='Filter by component name (optional)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the incident analyzer."""
    args = parse_args()
    
    try:
        # Initialize pipeline with default settings
        pipeline = LogProcessingPipeline(
            db_url=args.db_url,
            ollama_base_url="http://localhost:11434",  # Default Ollama URL
            model_name="mistral",  # Default model
            batch_size=100
        )
        
        # Initialize anomaly detector
        detector = IncidentAnomalyDetector(
            pipeline=pipeline,
            eps=args.eps,
            min_samples=args.min_samples
        )
        
        # Build additional filters
        filters = {}
        if args.level:
            filters['level'] = args.level
        if args.component:
            filters['component'] = args.component
            
        # Detect anomalies
        logger.info(f"Analyzing logs from the last {args.hours} hours...")
        anomalies_df = detector.detect_anomalies(
            table_name=args.table,
            hours=args.hours,
            additional_filters=filters
        )
        
        # Display results
        if anomalies_df.empty:
            logger.info("No anomalies detected")
        else:
            logger.info("\nTop anomalies:")
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(anomalies_df[
                ['timestamp', 'level', 'component', 'message', 
                 'is_embedding_anomaly', 'is_numeric_anomaly']
            ].head().to_string())
            
            if len(anomalies_df) > 5:
                logger.info(f"\n... and {len(anomalies_df) - 5} more anomalies")
                
        # Save to file if requested
        if args.output and not anomalies_df.empty:
            anomalies_df.to_csv(args.output, index=False)
            logger.info(f"Saved anomalies to {args.output}")
            
    except Exception as e:
        logger.error(f"Error analyzing logs: {str(e)}")
        raise
        
    finally:
        pipeline.close()

if __name__ == '__main__':
    main() 