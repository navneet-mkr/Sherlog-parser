"""Module for end-to-end log processing pipeline."""

import logging
import time
from typing import Optional
from pathlib import Path
import pandas as pd

from src.eval.datasets import DatasetLoader
from src.core.timeseries import LogTimeSeriesDB
from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer

logger = logging.getLogger(__name__)

class LogProcessingPipeline:
    """Pipeline for processing logs from ingestion to storage."""
    
    def __init__(
        self,
        db_url: str,
        ollama_base_url: str = "http://localhost:11434",
        model_name: str = "mistral",
        similarity_threshold: float = 0.8,
        batch_size: int = 100,
        cache_dir: Optional[str] = None
    ):
        """Initialize the pipeline.
        
        Args:
            db_url: TimescaleDB connection URL
            ollama_base_url: Base URL for Ollama service
            model_name: Name of the model to use
            similarity_threshold: Threshold for template matching
            batch_size: Number of logs to process in each batch
            cache_dir: Directory for caching parser results
        """
        self.db = LogTimeSeriesDB(db_url)
        self.ollama_analyzer = create_ollama_analyzer(
            base_url=ollama_base_url,
            model_id=model_name,
            config={"cache_dir": cache_dir} if cache_dir else {}
        )
        self.parser = LogParserLLM(
            ollama_client=self.ollama_analyzer,
            similarity_threshold=similarity_threshold
        )
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.dataset_loader = DatasetLoader()
    
    def process_dataset(
        self,
        system: str,
        dataset_type: str = "loghub_2k",
        table_name: Optional[str] = None
    ) -> None:
        """Process an entire dataset and store in TimescaleDB.
        
        Args:
            system: System name (e.g., 'Apache')
            dataset_type: Dataset type (e.g., 'loghub_2k')
            table_name: Target table name (defaults to system_logs)
        """
        if table_name is None:
            table_name = f"{system.lower()}_logs"
            
        try:
            # Load raw logs
            logger.info(f"Loading logs for {system} from {dataset_type}")
            logs_df = self.dataset_loader.load_logs(system, dataset_type)
            
            # Initialize database
            logger.info(f"Initializing database table: {table_name}")
            self.db.initialize_db(table_name)
            
            # Process in batches
            total_logs = len(logs_df)
            for i in range(0, total_logs, self.batch_size):
                batch_df = logs_df.iloc[i:i + self.batch_size]
                self._process_batch(batch_df, table_name)
                logger.info(f"Processed {min(i + self.batch_size, total_logs)}/{total_logs} logs")
                
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            raise
            
    def _process_batch(self, batch_df: pd.DataFrame, table_name: str) -> None:
        """Process a batch of logs and store in database.
        
        Args:
            batch_df: DataFrame containing batch of logs
            table_name: Target table name
        """
        try:
            # Prepare logs for parsing
            logs = batch_df['Content'].tolist()
            inference_times = []
            
            # Parse logs with timing
            start_time = time.time()
            logs_with_ids = [(str(i), msg) for i, msg in enumerate(logs)]
            parsed_results = self.parser.parse_logs_batch(logs_with_ids)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            logger.info(
                f"Parsed {len(logs)} logs in {inference_time:.2f}s "
                f"({len(logs)/inference_time:.2f} logs/s)"
            )
            
            # Extract templates and parameters
            templates = []
            parameters = []
            for template, params in parsed_results:
                templates.append(template)
                parameters.append(params)
            
            # Prepare for database
            prepared_df = self.db.prepare_log_data(batch_df)
            
            # Update with parsed results
            final_df = self.db.update_parsed_results(prepared_df, templates, parameters)
            
            # Store in database
            self.db.bulk_insert_logs(final_df, table_name)
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def query_logs(
        self,
        table_name: str,
        time_range: Optional[tuple] = None,
        level: Optional[str] = None,
        component: Optional[str] = None,
        template_pattern: Optional[str] = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """Query processed logs with filters.
        
        Args:
            table_name: Table to query
            time_range: Optional tuple of (start_time, end_time)
            level: Optional log level filter
            component: Optional component filter
            template_pattern: Optional template pattern to match
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching log entries
        """
        # Build query conditions
        conditions = []
        params = {}
        
        if time_range:
            conditions.append("timestamp BETWEEN %(start_time)s AND %(end_time)s")
            params.update({
                'start_time': time_range[0],
                'end_time': time_range[1]
            })
            
        if level:
            conditions.append("level = %(level)s")
            params['level'] = level
            
        if component:
            conditions.append("component = %(component)s")
            params['component'] = component
            
        if template_pattern:
            conditions.append("template LIKE %(template)s")
            params['template'] = f"%{template_pattern}%"
            
        # Construct query
        query = f"SELECT * FROM {table_name}"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        return self.db.query_logs(query, params)
    
    def close(self) -> None:
        """Clean up resources."""
        self.db.close() 