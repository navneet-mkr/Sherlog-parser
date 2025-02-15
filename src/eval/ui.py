"""Streamlit interface for log parser evaluation."""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Dict, Any, Type, Tuple, cast
from dataclasses import asdict
import logging
import os

from src.pathway_pipeline.eval_pipeline import EvaluationPipeline
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
        page_title="Log Parser Evaluation",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Log Parser Evaluation Dashboard")
    st.markdown("""
    This dashboard allows you to evaluate the log parsing pipeline against benchmark datasets.
    Select a dataset and system to begin evaluation.
    """)
    
    init_session_state()
    
    # Dataset selection
    system, dataset_type = show_dataset_selector()
    
    # Model configuration
    config = show_model_config()
    
    # Run evaluation
    if st.button("Run Evaluation"):
        logger.info(f"Starting evaluation for {system}/{dataset_type}")
        try:
            with st.spinner("Running evaluation..."):
                pipeline = EvaluationPipeline(
                    base_dir="./data/eval_datasets",
                    dataset_type=dataset_type,
                    system=system,
                    llm_api_base=OLLAMA_BASE_URL,
                    llm_model=OLLAMA_MODEL,
                    similarity_threshold=config["similarity_threshold"],
                    batch_size=config["batch_size"]
                )
                
                pipeline.setup_pipeline()
                metrics = pipeline.evaluate()
                
                # Load results
                results_dir = Path("./output/eval")
                templates_df = pd.read_csv(results_dir / f"{system}_{dataset_type}_templates.csv")
                
                st.session_state.results = {
                    "metrics": asdict(metrics),
                    "templates": templates_df
                }
                logger.info("Evaluation completed successfully")
        
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            return
    
    # Show results if available
    if st.session_state.results:
        show_results(
            st.session_state.results["metrics"],
            st.session_state.results["templates"]
        )

if __name__ == "__main__":
    main() 