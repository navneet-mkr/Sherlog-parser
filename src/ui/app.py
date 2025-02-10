"""Streamlit UI for log parsing application."""

import os
import logging
from pathlib import Path
from typing import Optional

import streamlit as st
from pydantic import BaseModel

from src.pathway_pipeline.pipeline import LogParsingPipeline, PipelineConfig
from src.ui.components import (
    show_file_uploader,
    show_log_viewer,
    show_template_viewer,
    show_sidebar_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

class UIConfig(BaseModel):
    """UI Configuration."""
    upload_dir: str = "./data/uploads"
    output_dir: str = "./output"
    cache_dir: str = "./cache"

def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "config" not in st.session_state:
        st.session_state.config = UIConfig()

def create_pipeline() -> None:
    """Create and initialize the pipeline."""
    try:
        config = PipelineConfig(
            input_dir=st.session_state.config.upload_dir,
            output_dir=st.session_state.config.output_dir,
            cache_dir=st.session_state.config.cache_dir,
            ollama_base_url=OLLAMA_BASE_URL,
            model_name=OLLAMA_MODEL
        )
        
        st.session_state.pipeline = LogParsingPipeline(config)
        st.session_state.pipeline.setup_pipeline()
        
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        st.error(f"Failed to initialize pipeline: {str(e)}")

def main():
    """Main UI application."""
    st.set_page_config(
        page_title="Log Parser AI",
        page_icon="🔍",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("Log Parser AI")
    st.markdown("""
    This application uses AI to parse and analyze log files.
    Upload your log files and the system will automatically:
    1. Extract log templates
    2. Identify variables
    3. Group similar log patterns
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # File uploader
    uploaded_files = show_file_uploader()
    
    if uploaded_files:
        # Save uploaded files
        upload_dir = Path(st.session_state.config.upload_dir)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        for file in uploaded_files:
            file_path = upload_dir / file.name
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            logger.info(f"Saved uploaded file: {file_path}")
        
        # Create pipeline if not exists
        if not st.session_state.pipeline:
            create_pipeline()
        
        # Run pipeline
        if st.session_state.pipeline:
            if st.button("Process Logs"):
                with st.spinner("Processing logs..."):
                    try:
                        st.session_state.pipeline.run()
                        st.success("Log processing complete!")
                    except Exception as e:
                        logger.error(f"Failed to process logs: {str(e)}")
                        st.error(f"Failed to process logs: {str(e)}")
        
        # Show results
        col1, col2 = st.columns(2)
        
        with col1:
            show_log_viewer(st.session_state.config.upload_dir)
        
        with col2:
            show_template_viewer(st.session_state.config.output_dir)
    
    # Show sidebar info
    show_sidebar_info()

if __name__ == "__main__":
    main() 