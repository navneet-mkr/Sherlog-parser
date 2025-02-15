"""
Metrics tracking and observability for log parsing.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ParsingMetrics:
    """Metrics for a single log parsing run."""
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_logs: int = 0
    processed_logs: int = 0
    unique_templates: int = 0
    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    debug_count: int = 0
    unknown_count: int = 0
    parsing_errors: List[Dict] = field(default_factory=list)
    template_stats: Dict[str, int] = field(default_factory=dict)
    processing_time_ms: float = 0.0
    avg_time_per_log_ms: float = 0.0
    
    def log_error(self, error: Exception, log_content: str, log_id: int) -> None:
        """Log a parsing error.
        
        Args:
            error: The exception that occurred
            log_content: The log line that caused the error
            log_id: ID of the log line
        """
        self.error_count += 1
        self.parsing_errors.append({
            'log_id': log_id,
            'content': log_content,
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        logger.error(f"Error parsing log {log_id}: {str(error)}")
    
    def update_event_counts(self, event_type: str) -> None:
        """Update event type counts.
        
        Args:
            event_type: Type of event (error, warning, info, etc.)
        """
        if event_type == 'error':
            self.error_count += 1
        elif event_type == 'warning':
            self.warning_count += 1
        elif event_type == 'info':
            self.info_count += 1
        elif event_type == 'debug':
            self.debug_count += 1
        else:
            self.unknown_count += 1
    
    def update_template_stats(self, template_id: str) -> None:
        """Update template occurrence counts.
        
        Args:
            template_id: ID of the template
        """
        if template_id not in self.template_stats:
            self.template_stats[template_id] = 0
            self.unique_templates += 1
        self.template_stats[template_id] += 1
    
    def finalize(self) -> None:
        """Finalize metrics after parsing is complete."""
        self.end_time = datetime.now()
        self.processing_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.avg_time_per_log_ms = self.processing_time_ms / self.total_logs if self.total_logs > 0 else 0
    
    def save(self, output_dir: Path) -> None:
        """Save metrics to file.
        
        Args:
            output_dir: Directory to save metrics
        """
        metrics_file = output_dir / "parsing_metrics.json"
        
        metrics_dict = {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_logs': self.total_logs,
            'processed_logs': self.processed_logs,
            'unique_templates': self.unique_templates,
            'event_counts': {
                'error': self.error_count,
                'warning': self.warning_count,
                'info': self.info_count,
                'debug': self.debug_count,
                'unknown': self.unknown_count
            },
            'template_stats': self.template_stats,
            'processing_time_ms': self.processing_time_ms,
            'avg_time_per_log_ms': self.avg_time_per_log_ms,
            'parsing_errors': self.parsing_errors
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        logger.info(f"Saved parsing metrics to {metrics_file}")
        
        # Log summary
        logger.info("Parsing Summary:")
        logger.info(f"  Total Logs: {self.total_logs}")
        logger.info(f"  Processed Successfully: {self.processed_logs}")
        logger.info(f"  Unique Templates: {self.unique_templates}")
        logger.info(f"  Processing Time: {self.processing_time_ms:.2f}ms")
        logger.info(f"  Avg Time per Log: {self.avg_time_per_log_ms:.2f}ms")
        logger.info("Event Counts:")
        logger.info(f"  Error: {self.error_count}")
        logger.info(f"  Warning: {self.warning_count}")
        logger.info(f"  Info: {self.info_count}")
        logger.info(f"  Debug: {self.debug_count}")
        logger.info(f"  Unknown: {self.unknown_count}")

class MetricsTracker:
    """Tracker for parsing metrics with timing context manager."""
    
    def __init__(self, output_dir: Path):
        """Initialize metrics tracker.
        
        Args:
            output_dir: Directory to save metrics
        """
        self.output_dir = output_dir
        self.metrics = ParsingMetrics()
        self._start_time = None
    
    def __enter__(self) -> ParsingMetrics:
        """Start tracking metrics.
        
        Returns:
            ParsingMetrics object
        """
        self._start_time = time.time()
        return self.metrics
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Finalize and save metrics."""
        if exc_type is None:
            self.metrics.finalize()
            self.metrics.save(self.output_dir)
        else:
            logger.error(f"Error during parsing: {str(exc_val)}")
            # Still save metrics even if there was an error
            self.metrics.finalize()
            self.metrics.save(self.output_dir) 