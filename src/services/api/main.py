"""Script for real-time log parsing using pre-trained patterns."""

import argparse
import logging
import sys
from pathlib import Path

from models import Settings
from log_parser import LogParser
from data_storage import DuckDBManager
from utils import setup_logging, validate_openai_key

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed argument namespace
    """
    parser = argparse.ArgumentParser(
        description="Parse log lines using pre-trained patterns."
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input log file or '-' for stdin"
    )
    
    parser.add_argument(
        "--db_path",
        type=str,
        help=f"Path to the DuckDB database with patterns (default: {Settings().db_path})"
    )
    
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main entry point for the log parsing script."""
    args = parse_args()
    
    # Setup logging
    setup_logging(getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)
    
    # Load settings
    settings = Settings()
    if args.db_path:
        settings.db_path = args.db_path
    
    # Validate OpenAI API key
    try:
        validate_openai_key()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    
    try:
        # Initialize components
        db = DuckDBManager(settings)
        parser = LogParser(settings, db)
        
        # Setup input source
        if args.input == '-':
            logger.info("Reading from stdin (press Ctrl+D to end)")
            input_lines = sys.stdin
        else:
            input_path = Path(args.input)
            if not input_path.is_file():
                logger.error(f"Input file not found: {args.input}")
                sys.exit(1)
            logger.info(f"Reading from file: {args.input}")
            input_lines = open(input_path, 'r')
        
        try:
            # Process lines
            for line in input_lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Parse the line
                parsed_log = parser.parse_line(line)
                
                # Store the result
                db.store_single_log(parsed_log)
                
                # Print the parsed fields
                if parsed_log.parsed_fields:
                    logger.info(f"Parsed fields: {parsed_log.parsed_fields}")
                else:
                    logger.warning(f"No pattern matched: {line}")
                    
        finally:
            if args.input != '-':
                input_lines.close()
        
    except Exception as e:
        logger.error(f"Error processing logs: {str(e)}")
        sys.exit(1)
    finally:
        db.close()

if __name__ == "__main__":
    main() 