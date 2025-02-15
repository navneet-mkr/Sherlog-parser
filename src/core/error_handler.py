"""Module providing error handling functionality."""

from functools import wraps
from typing import Type, Tuple, Callable, Any, Optional

class ErrorHandler:
    """Class for handling errors and exceptions."""
    
    def error_handler(self, reraise: bool = True, exclude: Optional[Tuple[Type[Exception], ...]] = None):
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
                    self.handle_error(e)
                    
                    if reraise:
                        raise
            return wrapper
        return decorator
    
    def handle_error(self, error: Exception) -> None:
        """Handle an error.
        
        Args:
            error: The exception to handle
        """
        # Basic error handling - can be extended as needed
        import logging
        logger = logging.getLogger(__name__)
        logger.error(
            "Error occurred",
            exc_info=error,
            extra={
                "error_code": getattr(error, "error_code", None),
                "details": getattr(error, "details", None)
            }
        )

# Create singleton instance
_handler = ErrorHandler()
error_handler = _handler.error_handler

__all__ = ['error_handler'] 