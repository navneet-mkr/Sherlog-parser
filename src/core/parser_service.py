"""
Service layer for log parsing operations.
"""

import logging
import time
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime
import pandas as pd

from src.models.log_parser import LogParserLLM
from src.models.ollama import create_ollama_analyzer
from src.core.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class ParserService:
    """Service layer for log parsing operations."""
    
    def __init__(
        self,
        llm_model: str = "ollama/mistral",
        llm_api_base: str = "http://localhost:11434",
        similarity_threshold: float = 0.8,
        batch_size: int = 32
    ):
        """Initialize parser service.
        
        Args:
            llm_model: Name of the LLM model to use
            llm_api_base: Base URL for LLM API
            similarity_threshold: Threshold for template matching
            batch_size: Number of logs to process in each batch
        """
        logger.info("Initializing ParserService")
        logger.info(f"Model: {llm_model}")
        logger.info(f"API base: {llm_api_base}")
        logger.info(f"Similarity threshold: {similarity_threshold}")
        logger.info(f"Batch size: {batch_size}")
        
        self.batch_size = batch_size
        
        # Initialize Ollama analyzer
        logger.info("Initializing Ollama analyzer")
        ollama_analyzer = create_ollama_analyzer(
            base_url=llm_api_base,
            model_id=llm_model.split('/')[-1],
            config={"temperature": 0.1}
        )
        
        # Initialize LogParserLLM
        logger.info("Initializing LogParserLLM")
        self.parser = LogParserLLM(
            ollama_client=ollama_analyzer,
            similarity_threshold=similarity_threshold
        )
        
    def parse_logs(
        self,
        log_file: str,
        output_dir: str,
        template_file: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Parse logs from a file.
        
        Args:
            log_file: Path to log file
            output_dir: Directory to save results
            template_file: Optional path to existing templates file
            
        Returns:
            Tuple of (parsed_logs DataFrame, templates DataFrame)
        """
        logger.info(f"Starting log parsing process for: {log_file}")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set up metrics tracking
        tracker = MetricsTracker(output_path)
        
        with tracker as metrics:
            # Read logs
            logger.info("Reading log file...")
            logs_df = self._read_logs(log_file)
            total_logs = len(logs_df)
            metrics.total_logs = total_logs
            logger.info(f"Successfully read {total_logs} logs")
            
            # Process logs
            processed_logs = []
            templates = {}
            
            # Calculate batch information
            num_batches = (total_logs + self.batch_size - 1) // self.batch_size
            logger.info(f"Processing {total_logs} logs in {num_batches} batches (batch_size={self.batch_size})")
            
            current_batch = 0
            for start_idx in range(0, total_logs, self.batch_size):
                current_batch += 1
                end_idx = min(start_idx + self.batch_size, total_logs)
                batch_size = end_idx - start_idx
                
                logger.info(f"Processing batch {current_batch}/{num_batches} (logs {start_idx+1}-{end_idx})")
                
                batch_start_time = time.time()
                batch_processed = 0
                batch_templates = 0
                
                for idx in range(start_idx, end_idx):
                    row = logs_df.iloc[idx]
                    try:
                        # Parse log using LogParserLLM
                        parse_start_time = time.time()
                        template, params = self.parser.parse_log(row['content'], idx)
                        parse_time_ms = (time.time() - parse_start_time) * 1000
                        
                        # Infer event type
                        event_type = self._infer_event_type(template)
                        metrics.update_event_counts(event_type)
                        
                        # Store parsed result
                        processed_logs.append({
                            'log_id': idx,
                            'content': row['content'],
                            'timestamp': row.get('timestamp', datetime.now().isoformat()),
                            'log_level': row.get('log_level', 'INFO'),
                            'source': row.get('source', 'unknown'),
                            'template': template,
                            'parameters': params,
                            'event_type': event_type,
                            'parse_time_ms': parse_time_ms
                        })
                        metrics.processed_logs += 1
                        batch_processed += 1
                        
                        # Update template stats
                        if template not in templates:
                            template_id = f"E{len(templates) + 1}"
                            templates[template] = {
                                'template_id': template_id,
                                'template': template,
                                'count': 1
                            }
                            metrics.update_template_stats(template_id)
                            batch_templates += 1
                        else:
                            templates[template]['count'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing log {idx}: {str(e)}")
                        metrics.log_error(e, row['content'], idx)
                        continue
                
                # Log batch statistics
                batch_time = time.time() - batch_start_time
                batch_rate = batch_processed / batch_time if batch_time > 0 else 0
                logger.info(
                    f"Batch {current_batch} complete: "
                    f"processed {batch_processed}/{batch_size} logs "
                    f"({batch_rate:.1f} logs/sec), "
                    f"found {batch_templates} new templates"
                )
                
                # Calculate and log overall progress
                progress = (current_batch / num_batches) * 100
                total_templates = len(templates)
                avg_time_per_log = metrics.processing_time_ms / metrics.processed_logs if metrics.processed_logs > 0 else 0
                
                logger.info(
                    f"Overall Progress: {progress:.1f}% "
                    f"({metrics.processed_logs}/{total_logs} logs, "
                    f"{total_templates} templates, "
                    f"{avg_time_per_log:.1f}ms/log)"
                )
            
            # Convert to DataFrames
            logger.info("Converting results to DataFrames...")
            parsed_logs_df = pd.DataFrame(processed_logs)
            templates_df = pd.DataFrame(list(templates.values()))
            
            # Save results
            logger.info("Saving results...")
            self._save_results(parsed_logs_df, templates_df, output_path)
            
            logger.info(
                f"Processing complete: "
                f"{len(processed_logs)}/{total_logs} logs processed into "
                f"{len(templates)} templates"
            )
            return parsed_logs_df, templates_df
    
    def _read_logs(self, log_file: str) -> pd.DataFrame:
        """Read logs from file.
        
        Args:
            log_file: Path to log file
            
        Returns:
            DataFrame with log entries
        """
        logger.info(f"Reading log file: {log_file}")
        try:
            # Try reading as CSV first
            logger.info("Attempting to read as CSV...")
            df = pd.read_csv(log_file)
            if 'content' not in df.columns:
                logger.info("No 'content' column found, assuming single column of log lines")
                # If no content column, assume single column of log lines
                df = pd.DataFrame({'content': df.iloc[:, 0]})
            logger.info(f"Successfully read {len(df)} logs from CSV")
            return df
        except Exception as e:
            logger.info(f"CSV reading failed ({str(e)}), attempting to read as text file")
            # If CSV fails, read as text file
            with open(log_file) as f:
                lines = f.readlines()
            df = pd.DataFrame({'content': [line.strip() for line in lines]})
            logger.info(f"Successfully read {len(df)} logs from text file")
            return df
    
    def _infer_event_type(self, template: str) -> str:
        """Infer event type from template.
        
        Args:
            template: Log template string
            
        Returns:
            Inferred event type
        """
        template_lower = template.lower()
        
        if any(word in template_lower for word in ['error', 'fail', 'exception']):
            event_type = 'error'
        elif any(word in template_lower for word in ['warn', 'warning']):
            event_type = 'warning'
        elif any(word in template_lower for word in ['info', 'status']):
            event_type = 'info'
        elif any(word in template_lower for word in ['debug']):
            event_type = 'debug'
        else:
            event_type = 'unknown'
            
        logger.debug(f"Inferred event type '{event_type}' for template: {template}")
        return event_type
    
    def _save_results(
        self,
        parsed_logs_df: pd.DataFrame,
        templates_df: pd.DataFrame,
        output_dir: Path
    ) -> None:
        """Save parsing results.
        
        Args:
            parsed_logs_df: DataFrame with parsed logs
            templates_df: DataFrame with templates
            output_dir: Directory to save results
        """
        # Save parsed logs
        parsed_logs_file = output_dir / "parsed_logs.csv"
        logger.info(f"Saving {len(parsed_logs_df)} parsed logs to {parsed_logs_file}")
        parsed_logs_df.to_csv(parsed_logs_file, index=False)
        
        # Save templates
        templates_file = output_dir / "templates.csv"
        logger.info(f"Saving {len(templates_df)} templates to {templates_file}")
        templates_df.to_csv(templates_file, index=False)
        
        # Generate summary report
        logger.info("Generating summary report...")
        report = f"""# Log Parsing Summary

## Statistics
- Total Logs: {len(parsed_logs_df)}
- Unique Templates: {len(templates_df)}

## Event Types
{parsed_logs_df['event_type'].value_counts().to_string()}

## Top Templates
{templates_df.nlargest(10, 'count')[['template_id', 'template', 'count']].to_string()}

## Performance
- Average Parse Time: {parsed_logs_df['parse_time_ms'].mean():.2f}ms
- Max Parse Time: {parsed_logs_df['parse_time_ms'].max():.2f}ms
- Min Parse Time: {parsed_logs_df['parse_time_ms'].min():.2f}ms
"""
        
        report_file = output_dir / "parsing_report.md"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Saved summary report to {report_file}") 