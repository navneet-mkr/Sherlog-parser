import os
import logging
import streamlit as st
import requests
from pathlib import Path
from dagster import DagsterInstance, JobDefinition
from src.core import log_processing_job
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from huggingface_hub import hf_hub_download
from typing import Optional, Dict, TypedDict, Literal, Union, Any, List
from src.models.config import (
    ModelInfo,
    LLMConfig,
    PipelineConfig,
    RunConfig,
    AVAILABLE_MODELS,
    OllamaSettings
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

# Initialize Ollama settings
OLLAMA_SETTINGS = OllamaSettings(
    host=os.getenv("OLLAMA_HOST", "http://localhost"),
    port=int(os.getenv("OLLAMA_PORT", "11434"))
)

def get_available_models() -> Dict[str, ModelInfo]:
    """Get list of available models from Ollama server."""
    try:
        response = requests.get(f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/tags")
        if response.status_code == 200:
            available_models = response.json().get("models", [])
            return {
                model["name"]: ModelInfo(
                    name=model["name"],
                    model_id=model["name"],
                    description=AVAILABLE_MODELS.get(model["name"], ModelInfo(
                        name=model["name"],
                        model_id=model["name"],
                        description="No description available"
                    )).description,
                    context_length=AVAILABLE_MODELS.get(model["name"], ModelInfo(
                        name=model["name"],
                        model_id=model["name"],
                        description="No description available"
                    )).context_length
                )
                for model in available_models
                if not model["name"].endswith(":latest")  # Filter out latest tags
            }
    except Exception as e:
        logger.error(f"Failed to fetch models from Ollama: {str(e)}")
        return AVAILABLE_MODELS

def get_llm(config: LLMConfig) -> Dict[str, Any]:
    """Get LLM instance based on configuration.
    
    Args:
        config: LLM configuration
        
    Returns:
        Dictionary with LLM instance and configuration
        
    Raises:
        RuntimeError: If model loading fails
    """
    try:
        base_url = f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}"
        
        llm = Ollama(
            base_url=base_url,
            model=config.model_id,
            temperature=config.temperature,
            num_predict=config.num_predict,
            top_k=config.top_k,
            top_p=config.top_p,
            repeat_penalty=config.repeat_penalty
        )
        
        return {
            "llm": llm,
            "config": {
                "name": config.model_id,
                "description": AVAILABLE_MODELS.get(config.model_id, ModelInfo(
                    name=config.model_id,
                    model_id=config.model_id,
                    description="Custom model"
                )).description
            }
        }
            
    except Exception as e:
        logger.error(f"Error initializing Ollama model: {str(e)}")
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

def handle_model_selection() -> Optional[Dict[str, Any]]:
    """Handle model selection in the UI."""
    st.subheader("🤖 Model Selection")
    
    # Get available models
    models = get_available_models()
    
    if not models:
        st.error("❌ No models available. Please check if Ollama service is running.")
        return None
    
    selected_model = st.selectbox(
        "Select Model",
        list(models.keys()),
        format_func=lambda x: models[x].name,
        help="Choose a model to use for log analysis"
    )
    
    if selected_model:
        model_info = models[selected_model]
        st.markdown(f"**Description**: {model_info.description}")
        st.markdown(f"**Context Length**: {model_info.context_length} tokens")
        
        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="Higher values make the output more random"
            )
            
            top_p = st.slider(
                "Top P",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Nucleus sampling threshold"
            )
            
            top_k = st.slider(
                "Top K",
                min_value=1,
                max_value=100,
                value=40,
                help="Limit the next token selection to K tokens"
            )
        
        try:
            config = LLMConfig(
                model_id=selected_model,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )
            llm_config = get_llm(config)
            st.success("✅ Model is ready to use")
            return llm_config
        except Exception as e:
            st.error(f"Failed to initialize model: {str(e)}")
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

def pull_model(model_id: str) -> bool:
    """Pull a model from Ollama.
    
    Args:
        model_id: ID of the model to pull
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with st.spinner(f"Pulling model {model_id}..."):
            response = requests.post(
                f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/pull",
                json={"name": model_id},
                stream=True
            )
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for line in response.iter_lines():
                if line:
                    data = line.decode('utf-8')
                    status_text.text(f"Downloading: {data}")
                    # Update progress if available in response
                    if '"completed"' in data:
                        progress_bar.progress(100)
                        break
            
            status_text.empty()
            return True
    except Exception as e:
        logger.error(f"Failed to pull model {model_id}: {str(e)}")
        return False

def delete_model(model_id: str) -> bool:
    """Delete a model from Ollama.
    
    Args:
        model_id: ID of the model to delete
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        response = requests.delete(
            f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/delete",
            json={"name": model_id}
        )
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {str(e)}")
        return False

def get_model_details(model_id: str) -> Optional[Dict]:
    """Get detailed information about a model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        Optional[Dict]: Model details if available
    """
    try:
        response = requests.post(
            f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/show",
            json={"name": model_id}
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        logger.error(f"Failed to get model details for {model_id}: {str(e)}")
        return None

def manage_models():
    """Model management interface in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.header("🛠️ Model Management")
    
    # Get current models
    current_models = get_available_models()
    
    # Model pulling section
    st.sidebar.subheader("📥 Pull New Model")
    new_model = st.sidebar.text_input(
        "Model Name",
        help="Enter the name of the model to pull (e.g., llama2, mistral, codellama)"
    )
    
    if st.sidebar.button("Pull Model", help="Download the specified model"):
        if new_model:
            if pull_model(new_model):
                st.sidebar.success(f"✅ Successfully pulled {new_model}")
            else:
                st.sidebar.error(f"❌ Failed to pull {new_model}")
    
    # Model management section
    if current_models:
        st.sidebar.subheader("💾 Installed Models")
        for model_id, info in current_models.items():
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(f"**{info.name}**")
                
                # Show model details in expander
                with st.expander("Details"):
                    details = get_model_details(model_id)
                    if details:
                        st.json(details)
            
            with col2:
                if st.button("🗑️", key=f"delete_{model_id}", help=f"Delete {model_id}"):
                    if delete_model(model_id):
                        st.success(f"✅ Deleted {model_id}")
                    else:
                        st.error(f"❌ Failed to delete {model_id}")

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

    # Add model management to sidebar
    manage_models()
    
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