"""Streamlit interface for log analysis and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any
from cachetools import TTLCache, cached
from datetime import timedelta
import logging
import json

from src.models.config import Settings
from src.core.parser_service import ParserService
from src.core.errors import (
    error_handler,
    FileError,
    ParsingError,
    ClusteringError
)
from src.core.logging_config import setup_logging

# Set up logging
setup_logging(log_level="INFO", log_file="logs/ui.log")
logger = logging.getLogger(__name__)

# Initialize caches
CLUSTER_CACHE = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

# Initialize session state
if 'parser' not in st.session_state:
    st.session_state.parser = None
if 'config' not in st.session_state:
    st.session_state.config = None

@error_handler(
    reraise=False
)
def initialize_parser() -> None:
    """Initialize the log parser."""
    logger.info("Initializing parser")
    settings = Settings()
    st.session_state.config = settings
    st.session_state.parser = ParserService(
        llm_api_base=settings.ollama_base_url,
        llm_model=settings.model_name
    )
    logger.info("Parser initialized successfully")

@error_handler(
    reraise=False
)
def process_uploaded_file(uploaded_file) -> None:
    """Process uploaded log file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Raises:
        FileError: If file processing fails
    """
    if not st.session_state.parser:
        initialize_parser()
        
    with st.spinner("Processing log file..."):
        try:
            # Save uploaded file
            file_path = Path(st.session_state.config.upload_dir) / uploaded_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the file
            parsed_logs_df, templates_df = st.session_state.parser.parse_logs(
                log_file=str(file_path),
                output_dir=str(st.session_state.config.output_dir)
            )
            
            # Show success message
            n_templates = len(templates_df)
            st.success(
                f"Successfully processed {len(parsed_logs_df)} log lines into "
                f"{n_templates} templates"
            )
            
        except Exception as e:
            raise FileError(
                "Failed to process log file",
                details={
                    "file_name": uploaded_file.name,
                    "error": str(e)
                }
            )

def show_metrics() -> None:
    """Display parsing metrics."""
    output_dir = Path(st.session_state.config.output_dir)
    metrics_file = output_dir / "parsing_metrics.json"
    
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        st.header("Parsing Metrics")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Logs", metrics['total_logs'])
        with col2:
            st.metric("Processed", metrics['processed_logs'])
        with col3:
            st.metric("Templates", metrics['unique_templates'])
        with col4:
            st.metric("Avg Time/Log", f"{metrics['avg_time_per_log_ms']:.1f}ms")
        
        # Event type distribution
        st.subheader("Event Type Distribution")
        event_counts = metrics['event_counts']
        fig = px.pie(
            values=list(event_counts.values()),
            names=list(event_counts.keys()),
            title="Event Types"
        )
        st.plotly_chart(fig)
        
        # Show errors if any
        if metrics['parsing_errors']:
            with st.expander("Parsing Errors"):
                st.write(f"Total Errors: {len(metrics['parsing_errors'])}")
                for error in metrics['parsing_errors']:
                    st.error(
                        f"Log ID: {error['log_id']}\n"
                        f"Content: {error['content']}\n"
                        f"Error: {error['error']}"
                    )

@error_handler(
    reraise=False
)
def display_template_info(template_id: str) -> None:
    """Display information about a specific template.
    
    Args:
        template_id: Template identifier
        
    Raises:
        ClusteringError: If template information cannot be displayed
    """
    try:
        # Read results
        output_dir = Path(st.session_state.config.output_dir)
        results_file = output_dir / "parsed_logs.csv"
        templates_file = output_dir / "templates.csv"
        
        if results_file.exists() and templates_file.exists():
            results_df = pd.read_csv(results_file)
            templates_df = pd.read_csv(templates_file)
            
            # Get template data
            template_logs = results_df[results_df['template_id'] == template_id]
            template_info = templates_df[templates_df['template_id'] == template_id].iloc[0]
            
            st.subheader(f"Template {template_id}")
            
            # Display template statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Log Count", len(template_logs))
            with col2:
                st.metric("Template", template_info['template'])
            with col3:
                avg_time = template_logs['parse_time_ms'].mean()
                st.metric("Avg Parse Time", f"{avg_time:.1f}ms")
            
            # Show template pattern
            st.code(template_info['template'], language="text")
            
            # Show sample log lines
            with st.expander("Sample Log Lines"):
                for _, row in template_logs.head(10).iterrows():
                    st.text(row['content'])
                    
            # Show variable distribution if available
            if 'parameters' in template_logs.columns:
                with st.expander("Variable Analysis"):
                    st.write("Variable Distributions:")
                    for _, row in template_logs.head(5).iterrows():
                        if row['parameters']:
                            st.json(row['parameters'])
                
    except Exception as e:
        raise ClusteringError(
            "Failed to display template information",
            details={
                "template_id": template_id,
                "error": str(e)
            }
        )

def main():
    """Main Streamlit application."""
    st.title("Sherlog-parser")
    st.write("Upload a log file to analyze patterns and extract templates.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a log file",
        type=["log", "txt"],
        help="Upload a log file to analyze"
    )
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)
        
        # Display results if available
        if st.session_state.parser:
            output_dir = Path(st.session_state.config.output_dir)
            results_file = output_dir / "parsed_logs.csv"
            templates_file = output_dir / "templates.csv"
            
            if results_file.exists() and templates_file.exists():
                # Show metrics
                show_metrics()
                
                # Read results
                results_df = pd.read_csv(results_file)
                templates_df = pd.read_csv(templates_file)
                
                # Template selection
                template_ids = templates_df['template_id'].tolist()
                selected_template = st.selectbox(
                    "Select template to view",
                    template_ids,
                    format_func=lambda x: f"Template {x}"
                )
                
                if selected_template:
                    display_template_info(selected_template)
                    
                # Show template distribution
                st.subheader("Template Distribution")
                template_counts = results_df['template_id'].value_counts()
                fig = px.bar(
                    x=template_counts.index,
                    y=template_counts.values,
                    labels={'x': 'Template ID', 'y': 'Count'}
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main() 