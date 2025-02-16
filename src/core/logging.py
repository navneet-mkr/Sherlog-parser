"""Logging configuration for log parser."""

import logging
import sys
from typing import Any, Callable
import time
from functools import wraps

import structlog
from structlog.types import Processor
from structlog.stdlib import ProcessorFormatter
from structlog.processors import CallsiteParameter

def setup_logging(level: str = "INFO", enable_debug: bool = False):
    """Setup structured logging using structlog.
    
    Args:
        level: Logging level
        enable_debug: Whether to enable debug logging
    """
    # Setup structlog processors
    shared_processors: list[Processor] = [
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if enable_debug:
        shared_processors.append(structlog.processors.CallsiteParameterAdder(
            parameters={
                CallsiteParameter.FUNC_NAME,
                CallsiteParameter.LINENO,
                CallsiteParameter.MODULE,
            }
        ))
    
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Setup standard library logging
    formatter = ProcessorFormatter(
        processor=structlog.processors.JSONRenderer(),
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    
    if enable_debug:
        debug_handler = logging.FileHandler('debug.log')
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(debug_handler)

def get_logger(name: str) -> structlog.BoundLogger:
    """Get a logger instance.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)

def log_duration(logger: structlog.BoundLogger) -> Callable:
    """Decorator to log function duration.
    
    Args:
        logger: Logger instance to use
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration_ms = (time.time() - start) * 1000
                logger.info(
                    "function_completed",
                    function=func.__name__,
                    duration_ms=duration_ms
                )
        return wrapper
    return decorator

def with_context(**context: Any) -> Callable:
    """Decorator to add context to log entries.
    
    Args:
        **context: Context key-value pairs
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = structlog.get_logger()
            with logger.bind(**context):
                return func(*args, **kwargs)
        return wrapper
    return decorator 