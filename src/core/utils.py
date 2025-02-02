"""Utility functions for the log parsing system."""

import logging
import os
from typing import List, Iterator, TextIO
from rich.logging import RichHandler
from rich.console import Console
from rich.traceback import install
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Install rich traceback handler
install(show_locals=True)

# Initialize rich console
console = Console()

def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration with rich formatting.
    
    Args:
        level: The logging level to use (default: logging.INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=True
        )]
    )

def chunk_file(file_obj: TextIO, chunk_size: int) -> Iterator[List[str]]:
    """Read a file in chunks of specified size.
    
    Args:
        file_obj: File object to read from
        chunk_size: Number of lines to read at once
    
    Yields:
        List of strings representing the lines in the current chunk
    """
    current_chunk = []
    for line in file_obj:
        current_chunk.append(line.strip())
        if len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = []
    
    if current_chunk:  # Don't forget the last chunk if it's smaller
        yield current_chunk

def validate_openai_key() -> None:
    """Validate that the OpenAI API key is set in environment variables."""
    if not os.getenv('OPENAI_API_KEY'):
        console.print(
            "[red]Error:[/red] OpenAI API key not found. "
            "Please set the [bold]OPENAI_API_KEY[/bold] environment variable."
        )
        raise ValueError(
            'OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.'
        )

def sanitize_column_name(name: str) -> str:
    """Convert a regex group name into a valid SQL column name.
    
    Args:
        name: The original name to sanitize
    
    Returns:
        A sanitized version of the name suitable for use as a SQL column
    """
    # Replace non-alphanumeric characters with underscores
    sanitized = ''.join(c if c.isalnum() else '_' for c in name)
    # Ensure it doesn't start with a number
    if sanitized[0].isdigit():
        sanitized = 'f_' + sanitized
    return sanitized.lower() 