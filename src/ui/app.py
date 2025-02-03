import os
import logging
import streamlit as st
from pathlib import Path
from dagster import DagsterInstance, JobDefinition
from src.core import log_processing_job
from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, TypedDict, Literal, Union, Any
from src.models.config import (
    ModelInfo,
    LLMConfig,
    PipelineConfig,
    RunConfig,
    AVAILABLE_MODELS
)

# Type definitions
class ModelInfo(TypedDict):
    name: str
    repo_id: str
    filename: str
    description: str
    context_length: int
    memory_required: str

class LLMConfig(TypedDict):
    llm: BaseLanguageModel
    config: Union[ModelInfo, Dict[str, str]]

ModelType = Literal["local", "api"]
OpenAIModel = Literal["gpt-3.5-turbo", "gpt-4"]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize constants
DAGSTER_HOST: str = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT: str = os.getenv("DAGSTER_PORT", "3000")
DAGSTER_GRPC_HOST: str = os.getenv("DAGSTER_GRPC_HOST", "localhost")
DAGSTER_GRPC_PORT: str = os.getenv("DAGSTER_GRPC_PORT", "4000")
DATA_DIR: Path = Path("/data/logs")
MODELS_DIR: Path = Path("/data/models")

def get_llm(config: LLMConfig) -> Dict[str, Any]:
    """Get LLM instance based on configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        Dictionary with LLM instance and configuration
        
    Raises:
        RuntimeError: If model loading fails
        ValueError: If API key is missing for API mode
    """
    if config.model_type == "local":
        model_info = AVAILABLE_MODELS[config.model_name]
        
        try:
            model_path = hf_hub_download(
                repo_id=model_info.repo_id,
                filename=model_info.filename,
                cache_dir=str(MODELS_DIR),
                resume_download=True
            )
            
            llm = LlamaCpp(
                model_path=model_path,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                n_ctx=model_info.context_length,
                verbose=True
            )
            
            return {
                "llm": llm,
                "config": model_info.dict()
            }
            
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
    else:  # API mode
        if not config.api_key:
            raise ValueError("API key required for OpenAI models")
            
        llm = ChatOpenAI(
            api_key=config.api_key,
            model_name=config.model_name,
            temperature=config.temperature
        )
        
        return {
            "llm": llm,
            "config": {
                "name": config.model_name,
                "description": "OpenAI's model optimized for chat"
            }
        }

def handle_model_selection() -> Optional[Dict[str, Any]]:
    """Handle model selection in the UI."""
    st.subheader("🤖 Model Selection")
    
    model_type = st.radio(
        "Select Model Type",
        ["local", "api"],
        format_func=lambda x: "Local Model (llama.cpp)" if x == "local" else "API (OpenAI)",
        help="Choose between running models locally or using cloud APIs"
    )
    
    if model_type == "local":
        selected_model = st.selectbox(
            "Select Model",
            list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x].name,
            help="Choose a model to use for log analysis"
        )
        
        if selected_model:
            model_info = AVAILABLE_MODELS[selected_model]
            st.markdown(f"**Description**: {model_info.description}")
            st.markdown(f"**Memory Required**: {model_info.memory_required}")
            st.markdown(f"**Context Length**: {model_info.context_length} tokens")
            
            try:
                config = LLMConfig(
                    model_type="local",
                    model_name=selected_model
                )
                llm_config = get_llm(config)
                st.success("✅ Model is ready to use")
                return llm_config
            except Exception as e:
                st.error(f"Failed to load model: {str(e)}")
                return None
                
    else:  # API mode
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key"
        )
        
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key to use API mode")
            return None
            
        model_name = st.selectbox(
            "Select OpenAI Model",
            ["gpt-3.5-turbo", "gpt-4"],
            help="Choose which OpenAI model to use"
        )
        
        try:
            config = LLMConfig(
                model_type="api",
                model_name=model_name,
                api_key=api_key
            )
            llm_config = get_llm(config)
            st.success("✅ API connection successful")
            return llm_config
        except Exception as e:
            st.error(f"Failed to initialize API: {str(e)}")
            return None

def init_dagster() -> DagsterInstance:
    """Initialize Dagster instance.
    
    Returns:
        DagsterInstance: The Dagster instance
        
    Raises:
        RuntimeError: If Dagster initialization fails
    """
    try:
        # Try to get the default instance first
        instance = DagsterInstance.get()
        logger.info("Using configured Dagster instance")
        return instance
    except Exception as e:
        logger.warning(f"Failed to get default Dagster instance: {str(e)}")
        try:
            # Fall back to ephemeral instance
            logger.info("Falling back to ephemeral Dagster instance")
            instance = DagsterInstance.ephemeral()
            return instance
        except Exception as e:
            logger.error(f"Failed to create ephemeral Dagster instance: {str(e)}")
            raise RuntimeError(f"Failed to initialize Dagster: {str(e)}")

def run_pipeline(config: PipelineConfig) -> str:
    """Run the log processing pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        str: Pipeline run ID
        
    Raises:
        RuntimeError: If pipeline execution fails
    """
    try:
        instance = init_dagster()
        
        run_config = RunConfig(
            ops={
                "read_log_file": {
                    "config": {
                        "file_path": str(config.file_path),
                        "encoding": config.encoding
                    }
                },
                "cluster_logs": {
                    "config": {"n_clusters": config.n_clusters}
                },
                "generate_embeddings": {
                    "config": {"batch_size": config.batch_size}
                }
            },
            resources={
                "llm": {
                    "config": config.llm_config.dict()
                }
            }
        )
        
        pipeline_run = instance.create_run(
            pipeline_name="log_processing_job",
            run_config=run_config.dict(),
            pipeline_code_origin=log_processing_job.get_python_origin(),
        )
        
        instance.launch_run(pipeline_run.run_id)
        return pipeline_run.run_id
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise RuntimeError(f"Pipeline execution failed: {str(e)}")

def save_uploaded_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Path:
    """Save uploaded file to disk.
    
    Args:
        uploaded_file: Streamlit UploadedFile
        
    Returns:
        Path: Path where file was saved
        
    Raises:
        RuntimeError: If file saving fails
    """
    try:
        # Ensure data directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {str(e)}")
        raise RuntimeError(f"Failed to save uploaded file: {str(e)}")

# Set page config
st.set_page_config(
    page_title="Sherlog Parser",
    page_icon="🔍",
    layout="wide"
)

# Main app
try:
    st.title("🔍 Sherlog Parser")
    st.write("Upload your log file and analyze patterns using our advanced ML pipeline.")

    # Model selection
    model_config = handle_model_selection()
    
    # Only show file upload if model is ready
    if (model_config["model_type"] == "local" and model_config["model_path"]) or \
       (model_config["model_type"] == "api" and model_config["api_key"]):
        
        # File uploader with additional information
        st.info("📁 Supported file types: .log and .txt files")
        uploaded_file = st.file_uploader(
            "Choose a log file",
            type=["log", "txt"],
            help="Upload a log file to analyze patterns and clusters"
        )

        # Parameters in columns
        col1, col2 = st.columns(2)
        with col1:
            num_clusters = st.slider(
                "Number of clusters",
                min_value=5,
                max_value=50,
                value=20,
                help="Adjust the number of pattern clusters to generate"
            )
        with col2:
            batch_size = st.slider(
                "Batch size",
                min_value=100,
                max_value=5000,
                value=1000,
                help="Adjust the processing batch size (larger = faster but more memory)"
            )

        if uploaded_file is not None:
            if st.button("🚀 Analyze Logs", help="Start the log analysis pipeline"):
                try:
                    with st.spinner("Processing logs..."):
                        # Save the uploaded file
                        file_path = save_uploaded_file(uploaded_file)
                        st.success(f"✅ File saved successfully: {file_path.name}")
                        
                        # Add model configuration to run config
                        config = PipelineConfig(
                            file_path=file_path,
                            encoding="utf-8",
                            n_clusters=num_clusters,
                            batch_size=batch_size
                        )
                        config.llm_config = model_config
                        
                        # Run the pipeline
                        run_id = run_pipeline(config)
                        
                        st.success("🎉 Pipeline started successfully!")
                        st.markdown(f"""
                        ### 📊 View Results
                        1. Open the [Dagster UI](http://{DAGSTER_HOST}:{DAGSTER_PORT})
                        2. Click on "Runs" in the left sidebar
                        3. Find your run with ID: `{run_id}`
                        4. View the pipeline progress and results
                        """)
                except RuntimeError as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Pipeline execution failed: {str(e)}")
                except Exception as e:
                    st.error("❌ An unexpected error occurred. Please try again.")
                    logger.error(f"Unexpected error: {str(e)}")

    # Sidebar information
    with st.sidebar:
        st.header("📖 Pipeline Steps")
        st.markdown("""
        1. **📥 Read Log File**
           - Loads and validates the log file
           - Performs initial data cleaning
        
        2. **🧮 Generate Embeddings**
           - Creates vector embeddings for log lines
           - Uses state-of-the-art language models
        
        3. **🎯 Cluster Logs**
           - Groups similar log patterns
           - Identifies common message types
        
        4. **📊 Analyze Patterns**
           - Extracts and validates patterns
           - Generates statistical insights
        """)

        st.header("💡 Tips")
        st.markdown("""
        - Adjust clusters based on log variety
        - Use larger batch sizes for better performance
        - Monitor progress in the Dagster UI
        """)

        # Add link to Dagster UI
        st.markdown("---")
        st.markdown(f"[🔗 Open Dagster UI](http://{DAGSTER_HOST}:{DAGSTER_PORT})")

except Exception as e:
    st.error("❌ Application failed to start. Please contact support.")
    logger.critical(f"Application error: {str(e)}", exc_info=True) 