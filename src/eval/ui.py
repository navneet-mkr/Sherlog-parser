"""Streamlit UI for evaluation dashboard."""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import asdict

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.pathway_pipeline.eval_pipeline import EvaluationPipeline
from src.eval.datasets import DatasetLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")

def init_session_state():
    """Initialize session state variables."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "results" not in st.session_state:
        st.session_state.results = None
    if "dataset_loader" not in st.session_state:
        st.session_state.dataset_loader = DatasetLoader("./data/eval_datasets")

def show_dataset_selector() -> tuple[str, str]:
    """Show dataset selection widgets."""
    available_systems = ["Apache", "Hadoop", "Linux", "Zookeeper"]
    dataset_types = ["loghub_2k", "logpub"]
    
    col1, col2 = st.columns(2)
    
    with col1:
        system = st.selectbox(
            "Select System",
            options=available_systems,
            help="Choose the system whose logs you want to evaluate"
        )
    
    with col2:
        dataset_type = st.selectbox(
            "Select Dataset",
            options=dataset_types,
            help="Choose the dataset type for evaluation"
        )
    
    return system, dataset_type

def show_model_config() -> Dict:
    """Show model configuration widgets."""
    with st.expander("Model Configuration"):
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.05,
            help="Threshold for template matching similarity"
        )
        
        batch_size = st.number_input(
            "Batch Size",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100,
            help="Number of logs to process in each batch"
        )
        
        return {
            "similarity_threshold": similarity_threshold,
            "batch_size": batch_size
        }

def show_results(metrics: Dict, templates_df: pd.DataFrame):
    """Show evaluation results."""
    st.markdown("### 📊 Evaluation Results")
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Grouping Accuracy", f"{metrics['grouping_accuracy']:.2%}")
    with col2:
        st.metric("Parsing Accuracy", f"{metrics['parsing_accuracy']:.2%}")
    with col3:
        st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    with col4:
        st.metric("Processing Time", f"{metrics['avg_inference_time_ms']:.1f}ms")
    
    # Template analysis
    st.markdown("### 📋 Template Analysis")
    
    # Template distribution chart
    template_counts = templates_df['template'].value_counts()
    fig = px.bar(
        x=template_counts.index,
        y=template_counts.values,
        title="Template Distribution",
        labels={"x": "Template ID", "y": "Count"}
    )
    st.plotly_chart(fig)
    
    # Template details
    st.markdown("### 🔍 Template Details")
    st.dataframe(templates_df)

def main():
    """Main UI application."""
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
        
        except Exception as e:
            st.error(f"Evaluation failed: {str(e)}")
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            return
    
    # Show results if available
    if st.session_state.results:
        show_results(
            st.session_state.results["metrics"],
            st.session_state.results["templates"]
        )

if __name__ == "__main__":
    main() 