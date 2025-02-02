"""Module for centralized error handling."""

from typing import Optional, Any, Dict
from rich.console import Console
from functools import wraps
from typing import Type, Tuple, Callable, Any
import logging

# Initialize console for rich output
console = Console()

# Define error codes
ERROR_CODES = {
    'CONFIG_ERROR': 1000,
    'FILE_ERROR': 2000,
    'PARSING_ERROR': 3000,
    'CLUSTERING_ERROR': 4000,
    'DATABASE_ERROR': 5000,
    'API_ERROR': 6000,
    'VALIDATION_ERROR': 7000
}

class SherlogError(Exception):
    """Base exception class for Sherlog-parser."""
    
    def __init__(
        self,
        message: str,
        error_code: int,
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize error.
        
        Args:
            message: Error message
            error_code: Numeric error code
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}

class ConfigError(SherlogError):
    """Configuration-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['CONFIG_ERROR'], details)

class FileError(SherlogError):
    """File handling errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['FILE_ERROR'], details)

class ParsingError(SherlogError):
    """Log parsing errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['PARSING_ERROR'], details)

class ClusteringError(SherlogError):
    """Clustering-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['CLUSTERING_ERROR'], details)

class DatabaseError(SherlogError):
    """Database operation errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['DATABASE_ERROR'], details)

class APIError(SherlogError):
    """API-related errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['API_ERROR'], details)

class ValidationError(SherlogError):
    """Data validation errors."""
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, ERROR_CODES['VALIDATION_ERROR'], details)

def error_handler(reraise: bool = True, exclude: Tuple[Type[Exception], ...] = None):
    """Decorator for handling errors in functions.
    
    Args:
        reraise: Whether to reraise the exception after handling
        exclude: Tuple of exception types to exclude from handling
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if exclude and isinstance(e, exclude):
                    raise
                
                # Handle the error
                if isinstance(e, SherlogError):
                    console.print(f"[red]Error {e.error_code}:[/red] {str(e)}")
                    if e.details:
                        console.print("[yellow]Details:[/yellow]")
                        for key, value in e.details.items():
                            console.print(f"  [blue]{key}:[/blue] {value}")
                else:
                    console.print(f"[red]Unexpected Error:[/red] {str(e)}")
                
                # Log the error
                logger = logging.getLogger(__name__)
                logger.error(
                    "Error occurred",
                    exc_info=e,
                    extra={
                        "error_code": getattr(e, "error_code", None),
                        "details": getattr(e, "details", None)
                    }
                )
                
                if reraise:
                    raise
                
            return None
        return wrapper
    return decorator 