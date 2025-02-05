import os
import logging
import streamlit as st
import requests
import time
import json
from pathlib import Path
from dagster import DagsterInstance
from src.core import log_processing_job
from langchain_community.llms import Ollama
from langchain.schema.language_model import BaseLanguageModel
from typing import Optional, Dict, TypedDict, Literal, Union, Any, Tuple
from src.models.config import (
    ModelInfo,
    LLMConfig,
    PipelineConfig,
    RunConfig,
    AVAILABLE_MODELS,
    OllamaSettings
)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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

class OllamaConnectionError(Exception):
    """Custom exception for Ollama connection issues."""
    pass

@retry(
    retry=retry_if_exception_type(requests.exceptions.RequestException),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(5)
)
def check_ollama_connection() -> bool:
    """Check if Ollama service is accessible with retry logic.
    
    Returns:
        bool: True if connection successful, False otherwise
        
    Raises:
        OllamaConnectionError: If connection fails after retries
    """
    try:
        response = requests.get(
            f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/tags",
            timeout=5
        )
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to connect to Ollama: {str(e)}")
        raise OllamaConnectionError(f"Failed to connect to Ollama: {str(e)}")

def show_connection_error(error_msg: str, container=st):
    """Display a user-friendly connection error message."""
    container.error("🔌 Connection Issue")
    container.markdown(f"**Error**: {error_msg}")
    container.markdown("""
    **Troubleshooting Steps**:
    1. Check if the Ollama service is running
    2. Verify network connectivity
    3. Ensure Ollama is accessible at the configured host/port
    4. Check system resources and logs
    
    The application will automatically retry connecting...
    """)

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

def format_size(size_bytes):
    """Format size in bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def calculate_eta(completed, total, start_time):
    """Calculate estimated time remaining."""
    if completed == 0:
        return "calculating..."
    
    elapsed = time.time() - start_time
    rate = completed / elapsed  # bytes per second
    remaining_bytes = total - completed
    eta_seconds = remaining_bytes / rate if rate > 0 else 0
    
    if eta_seconds < 60:
        return f"{eta_seconds:.0f} seconds"
    elif eta_seconds < 3600:
        return f"{eta_seconds/60:.0f} minutes"
    else:
        return f"{eta_seconds/3600:.1f} hours"

class DownloadStats:
    """Class to track download statistics with moving average speed calculation."""
    def __init__(self, window_size=5):
        self.speeds = []
        self.window_size = window_size
        self.last_update = time.time()
        self.last_completed = 0
        self.start_time = time.time()

    def update(self, completed: int) -> float:
        """Update stats and return current speed in bytes/second."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        if elapsed > 0:
            speed = (completed - self.last_completed) / elapsed
            self.speeds.append(speed)
            # Keep only the last window_size speeds
            if len(self.speeds) > self.window_size:
                self.speeds.pop(0)
            self.last_update = current_time
            self.last_completed = completed
            return sum(self.speeds) / len(self.speeds)
        return 0

    def get_eta(self, completed: int, total: int) -> str:
        """Calculate ETA based on moving average speed."""
        if not self.speeds or completed == 0:
            return "calculating..."
        
        avg_speed = sum(self.speeds) / len(self.speeds)
        remaining_bytes = total - completed
        eta_seconds = remaining_bytes / avg_speed if avg_speed > 0 else 0
        
        if eta_seconds < 60:
            return f"{eta_seconds:.0f} seconds"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.0f} minutes"
        else:
            return f"{eta_seconds/3600:.1f} hours"

def show_download_error(container, model_name: str, error_msg: str = None):
    """Show download error with terminal command alternative."""
    container.error("### ❌ Download Failed")
    if error_msg:
        container.markdown(f"**Error**: {error_msg}")
    
    container.markdown("""
    #### 🔧 Alternative Download Method
    
    You can try downloading the model directly using the terminal command:
    ```bash
    ollama pull {model_name}
    ```
    
    After the download completes, click the 'Retry Connection' button in the UI.
    """.format(model_name=model_name))

def pull_model_with_progress(model_name: str, progress_container=st.sidebar) -> Tuple[bool, Dict[str, Any]]:
    """Pull a model from Ollama with progress tracking.
    
    Args:
        model_name: Name of the model to pull
        progress_container: Streamlit container to show progress in
        
    Returns:
        Tuple[bool, Dict[str, Any]]: (success, latest_progress)
    """
    try:
        # Create a single container for all progress information
        download_container = progress_container.container()
        with download_container:
            header = st.markdown(f"### 📥 Downloading {model_name}")
            progress_bar = st.progress(0)
            status_text = st.empty()
            details_text = st.empty()
            
            # Initialize download statistics
            stats = DownloadStats()
            current_digest = None
            downloading_started = False
            
            response = requests.post(
                f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/pull",
                json={"model": model_name, "stream": True},
                stream=True
            )
            
            if response.status_code == 200:
                latest_progress = None
                
                for line in response.iter_lines():
                    if line:
                        progress_data = json.loads(line.decode())
                        status = progress_data.get("status", "")
                        
                        # Handle different status messages
                        if status == "pulling manifest":
                            status_text.markdown("⏳ Pulling manifest...")
                            details_text.empty()
                            
                        elif status == "downloading":
                            downloading_started = True
                            digest = progress_data.get("digest", "")
                            
                            # Only update UI elements if we have progress information
                            if "completed" in progress_data and "total" in progress_data:
                                progress = progress_data.get("completed", 0)
                                total = progress_data.get("total", 100)
                                
                                # Show new digest being downloaded
                                if digest != current_digest:
                                    current_digest = digest
                                    status_text.markdown(f"⬇️ Downloading layer: `{digest[:12]}...`")
                                
                                # Calculate progress percentage
                                progress_pct = (progress / total) * 100 if total > 0 else 0
                                progress_bar.progress(int(progress_pct))
                                
                                # Calculate speed using moving average
                                speed = stats.update(progress)
                                
                                # Update details with all information
                                details_html = f"""
                                <div style="margin: 10px 0; font-size: 0.9em;">
                                    <div style="color: #666;">
                                        <span style="display: inline-block; width: 80px;">Size:</span>
                                        {format_size(progress)} / {format_size(total)} ({progress_pct:.1f}%)
                                    </div>
                                    <div style="color: #666;">
                                        <span style="display: inline-block; width: 80px;">Speed:</span>
                                        {format_size(speed)}/s
                                    </div>
                                    <div style="color: #666;">
                                        <span style="display: inline-block; width: 80px;">ETA:</span>
                                        {stats.get_eta(progress, total)}
                                    </div>
                                </div>
                                """
                                details_text.markdown(details_html, unsafe_allow_html=True)
                                
                                latest_progress = {
                                    "progress": progress,
                                    "total": total,
                                    "status": status,
                                    "progress_pct": progress_pct,
                                    "speed": speed,
                                    "digest": digest
                                }
                            else:
                                # If we don't have progress info yet
                                status_text.markdown(f"⏳ Preparing download for layer: `{digest[:12]}...`")
                                details_text.empty()
                                
                        elif status == "verifying sha256 digest":
                            status_text.markdown("🔍 Verifying downloaded files...")
                            details_text.empty()
                            progress_bar.progress(100)
                            
                        elif status == "writing manifest":
                            status_text.markdown("📝 Writing manifest...")
                            
                        elif status == "removing any unused layers":
                            status_text.markdown("🧹 Cleaning up...")
                            
                        elif status == "success":
                            header.markdown("### ✅ Download Complete!")
                            status_text.markdown(f"Successfully downloaded {model_name}")
                            details_text.empty()
                            progress_bar.empty()
                            return True, latest_progress
                        
                        else:
                            status_text.markdown(f"ℹ️ Status: {status}")
                
                # If we get here without a success status
                if downloading_started:
                    show_download_error(download_container, model_name, "Download did not complete successfully")
                else:
                    show_download_error(download_container, model_name, "Failed to start download")
                return False, latest_progress
                
            show_download_error(download_container, model_name, "Failed to connect to Ollama")
            return False, None
            
    except Exception as e:
        logger.error(f"Failed to download model {model_name}: {str(e)}")
        if progress_container:
            show_download_error(progress_container, model_name, str(e))
        return False, None

def pull_model(model_name: str) -> bool:
    """Simple wrapper for pull_model_with_progress that only returns success status."""
    success, _ = pull_model_with_progress(model_name)
    return success

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

def check_ollama_status() -> Dict[str, Any]:
    """Check Ollama service status with improved error handling."""
    try:
        # First check basic connectivity
        try:
            is_connected = check_ollama_connection()
            if not is_connected:
                return {
                    "is_ready": False,
                    "message": "Ollama service is not responding",
                    "models_available": [],
                    "is_downloading": False,
                    "download_status": {},
                    "connection_error": True
                }
        except OllamaConnectionError as e:
            show_connection_error(str(e))
            return {
                "is_ready": False,
                "message": str(e),
                "models_available": [],
                "is_downloading": False,
                "download_status": {},
                "connection_error": True
            }

        # Get available models
        response = requests.get(
            f"{OLLAMA_SETTINGS.host}:{OLLAMA_SETTINGS.port}/api/tags",
            timeout=5
        )
        
        if response.status_code != 200:
            show_connection_error("Ollama service returned unexpected response")
            return {
                "is_ready": False,
                "message": "Waiting for Ollama service to start...",
                "models_available": [],
                "is_downloading": False,
                "download_status": {},
                "connection_error": True
            }
        
        models_data = response.json().get("models", [])
        available_models = [m["name"] for m in models_data if not m["name"].endswith(":latest")]
        
        # Check if mistral is available
        if "mistral" in available_models:
            return {
                "is_ready": True,
                "message": "Ollama is ready with mistral model!",
                "models_available": available_models,
                "is_downloading": False,
                "download_status": {},
                "connection_error": False
            }
        
        # Try to pull mistral if not available
        try:
            success, progress = pull_model_with_progress("mistral", st)
            
            if success:
                return {
                    "is_ready": True,
                    "message": "Mistral model download complete!",
                    "models_available": available_models + ["mistral"],
                    "is_downloading": False,
                    "download_status": {},
                    "connection_error": False
                }
            elif progress:
                return {
                    "is_ready": False,
                    "message": "Downloading mistral model...",
                    "models_available": available_models,
                    "is_downloading": True,
                    "download_status": {
                        "current_model": "mistral",
                        "progress": progress.get("progress", 0),
                        "total": progress.get("total", 100),
                        "status": progress.get("status", "downloading")
                    },
                    "connection_error": False
                }
            
            return {
                "is_ready": False,
                "message": "Initiating mistral model download...",
                "models_available": available_models,
                "is_downloading": True,
                "download_status": {},
                "connection_error": False
            }
                
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            show_connection_error(f"Failed to download model: {str(e)}")
            return {
                "is_ready": False,
                "message": "Error downloading model",
                "models_available": available_models,
                "is_downloading": False,
                "download_status": {},
                "connection_error": True
            }
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Ollama: {str(e)}")
        show_connection_error(str(e))
        return {
            "is_ready": False,
            "message": "Waiting for Ollama service to start...",
            "models_available": [],
            "is_downloading": False,
            "download_status": {},
            "connection_error": True
        }

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


    # Create placeholders for progress display
    progress_text = st.empty()
    progress_bar = st.empty()
    status_text = st.empty()

    # Check Ollama status before proceeding with main UI
    ollama_status = check_ollama_status()
    
    if not ollama_status["is_ready"]:
        # Create a clean loading screen in main area
        st.markdown("### 🚀 Initializing System")
        st.info(ollama_status["message"])
        
        if ollama_status.get("connection_error", False):
            # Add reconnection information
            st.markdown("""
            #### 🔄 Auto-Reconnection
            The application will automatically attempt to reconnect to the Ollama service.
            You can:
            - Wait for automatic reconnection
            - Check the Ollama service status
            - Verify your network connection
            """)
            # Add a manual retry button
            if st.button("🔄 Retry Connection"):
                st.rerun()
        elif ollama_status["is_downloading"]:
            st.markdown("#### Download Progress")
            download_status = ollama_status["download_status"]
            
            # Show current model being downloaded
            progress_text.text("Downloading: mistral")
            
            # Calculate and show progress
            progress = download_status.get("progress", 0)
            total = download_status.get("total", 100)
            progress_pct = (progress / total) * 100 if total > 0 else 0
            
            if progress > 0:  # Only show progress bar if we have actual progress
                # Show progress bar
                progress_bar.progress(int(progress_pct))
                
                # Show detailed status
                status_text.text(f"Progress: {progress:,} / {total:,} bytes ({progress_pct:.1f}%)")
            else:
                status_text.text("Preparing download...")
            
        # Add some helpful information while waiting
        with st.expander("ℹ️ What's happening?"):
            st.markdown("""
            The system is currently:
            1. Starting the Ollama service
            2. Downloading the Mistral model:
               - A powerful language model for log analysis
               - Optimized for pattern recognition
               - ~4GB download size
            3. Preparing the analysis pipeline
            
            This may take a few minutes on first startup. The model will be cached for future use.
            """)
            
        # Rerun the app every few seconds to check status
        time.sleep(2)  # Reduced sleep time for more responsive updates
        st.rerun()
        st.stop()

    # Clear progress display elements when done
    progress_text.empty()
    progress_bar.empty()
    status_text.empty()

    # Model selection
    model_config = handle_model_selection()
    
    if not model_config:
        st.error("❌ Application cannot start without available models.")
        st.info("Please ensure:")
        st.markdown("""
        - Ollama service is running
        - At least one model is loaded in Ollama
        - The application has network connectivity to Ollama
        """)
        st.stop()  # Stop the app here to prevent further errors
    
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