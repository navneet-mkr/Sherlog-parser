"""Streamlit interface for log analysis and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Type, Tuple, cast
from cachetools import TTLCache, cached
from datetime import timedelta

from src.models.config import Settings
from src.pathway_pipeline.pipeline import LogParsingPipeline, PipelineConfig
from src.core.errors import (
    error_handler,
    FileError,
    ParsingError,
    ClusteringError
)

# Initialize caches
CLUSTER_CACHE = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

# Define excluded exceptions with proper type casting
EXCLUDED_EXCEPTIONS: Tuple[Type[Exception], ...] = (cast(Type[Exception], KeyboardInterrupt), cast(Type[Exception], SystemExit))

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'config' not in st.session_state:
    st.session_state.config = None

@error_handler(
    reraise=False,
    exclude=EXCLUDED_EXCEPTIONS
)
def initialize_pipeline() -> None:
    """Initialize the log processing pipeline."""
    settings = Settings()
    config = PipelineConfig(
        input_dir=settings.upload_dir,
        output_dir=settings.output_dir,
        cache_dir=settings.cache_dir,
        ollama_base_url=settings.ollama_base_url,
        model_name=settings.model_name
    )
    st.session_state.config = config
    st.session_state.pipeline = LogParsingPipeline(
        log_dir=config.input_dir,
        output_dir=config.output_dir,
        cache_dir=config.cache_dir,
        llm_api_base=config.ollama_base_url,
        llm_model=config.model_name
    )

@error_handler(
    reraise=False,
    exclude=EXCLUDED_EXCEPTIONS
)
def process_uploaded_file(uploaded_file) -> None:
    """Process uploaded log file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Raises:
        FileError: If file processing fails
    """
    if not st.session_state.pipeline:
        initialize_pipeline()
        
    with st.spinner("Processing log file..."):
        try:
            # Save uploaded file
            file_path = Path(st.session_state.config.input_dir) / uploaded_file.name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Process the file
            st.session_state.pipeline.setup_pipeline()
            st.session_state.pipeline.run()
            
            # Show success message
            results_file = Path(st.session_state.config.output_dir) / "parsed_logs.csv"
            if results_file.exists():
                results_df = pd.read_csv(results_file)
                n_templates = len(pd.unique(results_df['template_id']))
                st.success(
                    f"Successfully processed {len(results_df)} log lines into "
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

@error_handler(
    reraise=False,
    exclude=EXCLUDED_EXCEPTIONS
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
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Log Count", len(template_logs))
            with col2:
                st.metric("Template", template_info['template'])
            
            # Show template pattern
            st.code(template_info['template'], language="text")
            
            # Show sample log lines
            with st.expander("Sample Log Lines"):
                for _, row in template_logs.head(10).iterrows():
                    st.text(row['content'])
                    
            # Show variable distribution if available
            if 'variables' in template_logs.columns:
                with st.expander("Variable Analysis"):
                    st.write("Variable Distributions:")
                    for _, row in template_logs.head(5).iterrows():
                        if row['variables']:
                            st.json(row['variables'])
                
    except Exception as e:
        raise ClusteringError(
            "Failed to display template information",
            details={
                "template_id": template_id,
                "error": str(e)
            }
        )

@error_handler(
    reraise=False,
    exclude=EXCLUDED_EXCEPTIONS
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
        if st.session_state.pipeline:
            output_dir = Path(st.session_state.config.output_dir)
            results_file = output_dir / "parsed_logs.csv"
            templates_file = output_dir / "templates.csv"
            
            if results_file.exists() and templates_file.exists():
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