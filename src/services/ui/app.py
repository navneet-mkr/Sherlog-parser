"""Streamlit interface for log analysis and visualization."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any
from cachetools import TTLCache, cached
from datetime import timedelta

from src.models.config import Settings
from src.core.pipeline import LogPipeline
from src.core.errors import (
    error_handler,
    FileError,
    ParsingError,
    ClusteringError
)

# Initialize caches
CLUSTER_CACHE = TTLCache(maxsize=100, ttl=timedelta(minutes=5).total_seconds())

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'clustering_state' not in st.session_state:
    st.session_state.clustering_state = None

@cached(CLUSTER_CACHE)
@error_handler(
    reraise=False,
    exclude=[KeyboardInterrupt, SystemExit]
)
def get_cluster_info(cluster_id: int) -> Dict[str, Any]:
    """Get cached cluster information.
    
    Args:
        cluster_id: Cluster identifier
        
    Returns:
        Dictionary containing cluster information
        
    Raises:
        ClusteringError: If cluster information cannot be retrieved
    """
    try:
        return st.session_state.clustering_state.get_cluster_info(cluster_id)
    except Exception as e:
        raise ClusteringError(
            f"Failed to get cluster information",
            details={
                "cluster_id": cluster_id,
                "error": str(e)
            }
        )

@error_handler(
    reraise=False,
    exclude=[KeyboardInterrupt, SystemExit]
)
def initialize_pipeline() -> None:
    """Initialize the log processing pipeline."""
    settings = Settings()
    st.session_state.pipeline = LogPipeline(settings=settings)

@error_handler(
    reraise=False,
    on_error=FileError,
    exclude=[KeyboardInterrupt, SystemExit]
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
            # Process the file
            clustering_state = st.session_state.pipeline.process_file(uploaded_file)
            st.session_state.clustering_state = clustering_state
            
            # Clear caches
            CLUSTER_CACHE.clear()
            
            # Show success message
            st.success(
                f"Successfully processed {clustering_state.n_samples} log lines into "
                f"{len(clustering_state.clusters)} clusters"
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
    on_error=ClusteringError,
    exclude=[KeyboardInterrupt, SystemExit]
)
def display_cluster_info(cluster_id: int) -> None:
    """Display information about a specific cluster.
    
    Args:
        cluster_id: Cluster identifier
        
    Raises:
        ClusteringError: If cluster information cannot be displayed
    """
    try:
        cluster_info = get_cluster_info(cluster_id)
        
        st.subheader(f"Cluster {cluster_id}")
        
        # Display cluster statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sample Count", cluster_info.size)
        with col2:
            if cluster_info.pattern:
                st.metric("Pattern Confidence", f"{cluster_info.pattern.confidence:.2%}")
        
        # Display pattern if available
        if cluster_info.pattern:
            st.code(cluster_info.pattern.pattern, language="python")
            
            # Show pattern matches
            with st.expander("Pattern Matches"):
                for line in cluster_info.sample_lines[:5]:
                    st.text(line)
        
        # Show sample lines
        with st.expander("Sample Log Lines"):
            for line in cluster_info.sample_lines[:10]:
                st.text(line)
                
    except Exception as e:
        raise ClusteringError(
            "Failed to display cluster information",
            details={
                "cluster_id": cluster_id,
                "error": str(e)
            }
        )

@error_handler(
    reraise=False,
    exclude=[KeyboardInterrupt, SystemExit]
)
def main():
    """Main Streamlit application."""
    st.title("Sherlog-parser")
    st.write("Upload a log file to analyze patterns and clusters.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a log file",
        type=["log", "txt"],
        help="Upload a log file to analyze"
    )
    
    if uploaded_file:
        process_uploaded_file(uploaded_file)
        
        if st.session_state.clustering_state:
            # Display cluster selection
            cluster_ids = list(st.session_state.clustering_state.clusters.keys())
            selected_cluster = st.selectbox(
                "Select cluster to view",
                cluster_ids,
                format_func=lambda x: f"Cluster {x}"
            )
            
            if selected_cluster is not None:
                display_cluster_info(selected_cluster)

if __name__ == "__main__":
    main() 