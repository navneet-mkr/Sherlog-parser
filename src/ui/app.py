import os
import logging
import streamlit as st
import time
from pathlib import Path
from dagster import DagsterInstance
from src.core import log_processing_job
from typing import Dict, Any
from src.models.config import (
    LLMConfig,
    PipelineConfig,
    RunConfig,
    OllamaSettings
)
from src.models.ollama_manager import OllamaManager
from src.models.file_handler import FileHandler
from src.ui.components import (
    show_connection_error,
    show_file_details,
    show_model_info,
    show_sidebar_info,
    show_model_settings
)

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
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost").rstrip('/')
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))

# Initialize managers
ollama_manager = OllamaManager(OllamaSettings(host=OLLAMA_HOST, port=OLLAMA_PORT))
file_handler = FileHandler(DATA_DIR)

def init_dagster() -> DagsterInstance:
    """Initialize Dagster instance."""
    try:
        instance = DagsterInstance.get()
        logger.info("Using configured Dagster instance")
        return instance
    except Exception as e:
        logger.warning(f"Failed to get default Dagster instance: {str(e)}")
        try:
            logger.info("Falling back to ephemeral Dagster instance")
            instance = DagsterInstance.ephemeral()
            return instance
        except Exception as e:
            logger.error(f"Failed to create ephemeral Dagster instance: {str(e)}")
            raise RuntimeError(f"Failed to initialize Dagster: {str(e)}")

def run_pipeline(config: PipelineConfig) -> str:
    """Run the log processing pipeline."""
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

def handle_model_selection() -> Dict[str, Any]:
    """Handle model selection in the UI."""
    st.subheader("🤖 Model Selection")
    
    # Get available models
    models = ollama_manager.get_available_models()
    
    if not models:
        st.error("❌ No models available. Please check if Ollama service is running.")
        return None
    
    # Create a display name for each model
    model_display = {
        model_id: f"{info['name']} ({info.get('details', {}).get('parameter_size', 'Unknown')})"
        for model_id, info in models.items()
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(models.keys()),
        format_func=lambda x: model_display.get(x, x),
        help="Choose a model to use for log analysis"
    )
    
    if selected_model:
        model_info = models[selected_model]
        show_model_info(model_info)
        
        # Get model settings
        settings = show_model_settings()
        
        try:
            # Create proper LLMConfig using the Pydantic model
            config = LLMConfig(
                model_id=selected_model.split(":")[0],  # Remove :latest tag
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                top_k=settings["top_k"],
                num_predict=2048,  # Using default from config
                repeat_penalty=1.1  # Using default from config
            )
            
            llm_config = {
                "model_type": "local",
                "model_path": None,  # Not needed for Ollama
                "config": config.model_dump()
            }
            st.success("✅ Model is ready to use")
            return llm_config
            
        except Exception as e:
            st.error(f"Failed to initialize model: {str(e)}")
            logger.error(f"Model initialization error: {str(e)}", exc_info=True)
            return None

def process_multiple_files(uploaded_files, llm_config: Dict[str, Any]):
    """Process multiple log files."""
    try:
        saved_files = file_handler.save_uploaded_files(uploaded_files)
        
        for file_info in saved_files:
            st.markdown(f"### Processing file: {file_info.name}")
            
            config = PipelineConfig(
                file_path=file_info.path,
                encoding="utf-8",
                n_clusters=5,
                batch_size=32,
                llm_config=llm_config
            )
            
            try:
                run_id = run_pipeline(config)
                st.success(f"✅ Started processing {file_info.name} - Run ID: {run_id}")
            except Exception as e:
                st.error(f"❌ Failed to process {file_info.name}: {str(e)}")
                
    except Exception as e:
        st.error(f"Failed to process files: {str(e)}")
        logger.error(f"Multiple file processing failed: {str(e)}", exc_info=True)

def show_model_selection_screen():
    """Show the model selection and file upload screen."""
    try:
        st.title("🔍 Log Analysis Pipeline")
        
        # Model selection
        model_config = handle_model_selection()
        
        if model_config:
            st.markdown("---")
            st.subheader("📁 Log Files")
            
            # File uploader with additional information
            st.info("📁 Supported file types: .log and .txt files")
            uploaded_files = st.file_uploader(
                "Choose log files",
                type=["log", "txt"],
                help="Upload log files to analyze patterns and clusters",
                accept_multiple_files=True
            )
            
            # Show file details if files are uploaded
            if uploaded_files:
                show_file_details(uploaded_files)
                
                # Pipeline configuration
                st.markdown("### ⚙️ Analysis Settings")
                num_clusters = st.slider(
                    "Number of clusters",
                    min_value=2,
                    max_value=10,
                    value=5,
                    help="Number of clusters to group similar log patterns"
                )
                
                if st.button("🚀 Analyze Logs", help="Start the log analysis pipeline"):
                    try:
                        with st.spinner("Initializing analysis..."):
                            # Process multiple files
                            process_multiple_files(uploaded_files, model_config)
                            
                            st.markdown("""
                            ### 📊 View Results
                            1. Open the [Dagster UI](http://{DAGSTER_HOST}:{DAGSTER_PORT})
                            2. Click on "Runs" in the left sidebar
                            3. View the pipeline progress and results for each file
                            """.format(DAGSTER_HOST=DAGSTER_HOST, DAGSTER_PORT=DAGSTER_PORT))
                            
                    except RuntimeError as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.info("👆 Upload one or more log files to begin analysis")
                
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        logger.error(f"Error in model selection screen: {str(e)}", exc_info=True)

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

    # Add auto-refresh for status checks
    st.markdown("""
        <style>
            div[data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0px;
            }
        </style>
    """, unsafe_allow_html=True)
    
    if 'last_check_time' not in st.session_state:
        st.session_state.last_check_time = time.time()
    
    current_time = time.time()
    time_since_last_check = current_time - st.session_state.last_check_time
    
    # Update last check time
    if time_since_last_check >= 10:  # Check every 10 seconds
        st.session_state.last_check_time = current_time
        st.rerun()

    # Create placeholders for progress display
    progress_text = st.empty()
    progress_bar = st.empty()
    status_text = st.empty()

    # Check Ollama status before proceeding with main UI
    ollama_status = ollama_manager.check_status()
    
    if not ollama_status["is_ready"]:
        # Create a clean loading screen in main area
        status_container = st.container()
        with status_container:
            st.markdown("### 🚀 Initializing System")
            st.info(ollama_status["message"])
            
            # Show last check time
            st.caption(f"Last status check: {time.strftime('%H:%M:%S')}")
            st.caption("Status is checked every 10 seconds automatically.")
        
        if ollama_status.get("connection_error", False):
            show_connection_error(ollama_status["message"])
            # Add a manual retry button
            if st.button("🔄 Retry Now"):
                st.session_state.last_check_time = 0  # Force immediate check
                st.rerun()
        elif ollama_status.get("needs_model", False):
            # Show model selection screen
            show_model_selection_screen()
        
        # Add some helpful information while waiting
        with st.expander("ℹ️ What's happening?"):
            st.markdown("""
            The system is currently:
            1. Starting the Ollama service
            2. Checking for available models
            3. Preparing the analysis pipeline
            
            The status is automatically checked every 10 seconds.
            You can also click 'Retry Now' to check immediately.
            """)
        
        st.stop()  # Stop here to prevent the rest of the UI from loading

    # Clear progress display elements when done
    progress_text.empty()
    progress_bar.empty()
    status_text.empty()

    # Show main UI
    show_model_selection_screen()
    
    # Show sidebar information
    show_sidebar_info(DAGSTER_HOST, DAGSTER_PORT)

except Exception as e:
    st.error("❌ Application failed to start. Please contact support.")
    logger.critical(f"Application error: {str(e)}", exc_info=True) 