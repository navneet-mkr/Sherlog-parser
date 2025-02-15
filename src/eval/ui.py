"""Streamlit interface for log parser evaluation."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict
import logging
import os
import json

from src.core.eval import Evaluator
from src.core.logging_config import setup_logging

# Set up logging
setup_logging(log_level="INFO", log_file="logs/eval.log")
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

def init_session_state():
    """Initialize session state variables."""
    if 'results' not in st.session_state:
        st.session_state.results = None
    logger.debug("Session state initialized")

def show_dataset_selector():
    """Show dataset selection widgets."""
    logger.debug("Showing dataset selector")
    system = st.selectbox(
        "Select system",
        ["Apache", "Hadoop", "HDFS", "Linux", "OpenStack", "Spark"]
    )
    
    dataset_type = st.selectbox(
        "Select dataset type",
        ["loghub_2k", "loghub_all"]
    )
    
    logger.info(f"Selected dataset: {system}/{dataset_type}")
    return system, dataset_type

def show_model_config():
    """Show model configuration widgets."""
    logger.debug("Showing model configuration")
    with st.expander("Model Configuration"):
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=128,
            value=32
        )
        
        config = {
            "similarity_threshold": similarity_threshold,
            "batch_size": batch_size
        }
        logger.info(f"Model configuration: {config}")
        return config

def show_results(metrics: Dict[str, Any], templates_df: pd.DataFrame):
    """Show evaluation results."""
    logger.info("Displaying evaluation results")
    
    # Metrics
    st.header("Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grouping Accuracy", f"{metrics['grouping_accuracy']:.4f}")
    with col2:
        st.metric("Parsing Accuracy", f"{metrics['parsing_accuracy']:.4f}")
    with col3:
        st.metric("F1 Grouping", f"{metrics['f1_grouping_accuracy']:.4f}")
    with col4:
        st.metric("F1 Template", f"{metrics['f1_template_accuracy']:.4f}")
    
    # Template distribution
    st.header("Template Distribution")
    template_counts = templates_df['template'].value_counts()
    fig = px.bar(
        x=template_counts.index,
        y=template_counts.values,
        labels={'x': 'Template', 'y': 'Count'}
    )
    st.plotly_chart(fig)
    
    # Template details
    st.header("Template Details")
    st.dataframe(templates_df)
    
    logger.info("Results display complete")

def main():
    """Main UI application."""
    logger.info("Starting evaluation UI")
    st.set_page_config(
        page_title="Log Parser Evaluation Results",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Log Parser Evaluation Results")
    st.markdown("""
    This dashboard displays the results of log parsing evaluations against benchmark datasets.
    To run new evaluations, use the command line:
    ```bash
    ./evaluate.sh [options]
    ```
    """)
    
    # Initialize evaluator
    try:
        logger.info("Initializing evaluator")
        evaluator = Evaluator(
            base_dir="./data/eval_datasets",
            dataset_type="loghub_2k",
            system="Apache",
            llm_model=os.getenv("MODEL_NAME", "mistral"),
            llm_api_base=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            output_dir="./output/eval",
            cache_dir="./cache/eval"
        )
        
        # Run evaluation if no results exist
        dataset_name = f"Apache_loghub_2k"
        metrics_file = Path("./output/eval") / f"{dataset_name}_metrics.json"
        
        if not metrics_file.exists():
            logger.info("No existing results found, starting evaluation")
            with st.spinner("Running evaluation..."):
                metrics = evaluator.evaluate()
                st.success("Evaluation complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {str(e)}")
        st.error(f"Failed to initialize evaluator: {str(e)}")
        return
    
    # Dataset selection
    system = st.selectbox(
        "Select system",
        ["Apache", "Hadoop", "HDFS", "Linux", "OpenStack", "Spark"]
    )
    
    dataset_type = st.selectbox(
        "Select dataset type",
        ["loghub_2k", "loghub_all"]
    )
    
    # Load and show results if available
    results_dir = Path("./output/eval")
    dataset_name = f"{system}_{dataset_type}"
    metrics_file = results_dir / f"{dataset_name}_metrics.json"
    templates_file = results_dir / f"{dataset_name}_templates.csv"
    
    if metrics_file.exists() and templates_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
        templates_df = pd.read_csv(templates_file)
        
        show_results(metrics, templates_df)
    else:
        st.warning(f"""
        No evaluation results found for {dataset_name}.
        
        First, ensure you have the dataset downloaded:
        ```bash
        ./download_datasets.sh
        ```
        
        Then run the evaluation:
        ```bash
        ./evaluate.sh
        ```
        
        The results will appear here once the evaluation is complete.
        """)

if __name__ == "__main__":
    main() 