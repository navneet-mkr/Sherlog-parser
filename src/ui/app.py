import os
import logging
import streamlit as st
from pathlib import Path
from dagster import DagsterInstance
from dagster._core.workspace.context import WorkspaceRequestContext
from dagster._core.workspace.load_target import PythonFileTarget
from dagster._core.workspace.workspace import WorkspaceLoader
from dagster.core.errors import DagsterError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add health check endpoint
def check_health():
    try:
        # Check if we can initialize Dagster
        instance = DagsterInstance.get()
        workspace = WorkspaceLoader(
            yaml_path="/opt/dagster/dagster_home/workspace.yaml"
        ).load_workspace()
        return True
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return False

# Create a special endpoint for health checks
if 'healthz' in st.experimental_get_query_params():
    if check_health():
        st.success("Healthy")
        st.stop()
    else:
        st.error("Unhealthy")
        st.stop()

# Initialize Dagster client
DAGSTER_HOST = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT = os.getenv("DAGSTER_PORT", "3000")
DATA_DIR = Path("/data/logs")

def init_dagster():
    """Initialize Dagster instance and workspace.
    
    Returns:
        tuple: (DagsterInstance, Workspace)
        
    Raises:
        RuntimeError: If Dagster initialization fails
    """
    try:
        instance = DagsterInstance.get()
        workspace = WorkspaceLoader(
            yaml_path="/opt/dagster/dagster_home/workspace.yaml"
        ).load_workspace()
        return instance, workspace
    except Exception as e:
        logger.error(f"Failed to initialize Dagster: {str(e)}")
        raise RuntimeError(f"Failed to initialize Dagster: {str(e)}")

def run_pipeline(file_path: str, num_clusters: int = 20, batch_size: int = 1000) -> str:
    """Run the log processing pipeline.
    
    Args:
        file_path: Path to the log file
        num_clusters: Number of clusters to generate
        batch_size: Batch size for processing
        
    Returns:
        str: Pipeline run ID
        
    Raises:
        RuntimeError: If pipeline execution fails
    """
    try:
        instance, workspace = init_dagster()
        context = WorkspaceRequestContext(instance, workspace)
        
        # Get the log processing pipeline
        try:
            location = next(iter(workspace.get_locations()))
            repository = next(iter(location.get_repositories().values()))
            pipeline = repository.get_job("log_processing_pipeline")
        except StopIteration:
            raise RuntimeError("Could not find log processing pipeline in workspace")
        
        # Configure and launch the pipeline
        run_config = {
            "ops": {
                "read_log_file": {
                    "config": {"encoding": "utf-8"},
                    "inputs": {"file_path": file_path}
                },
                "cluster_logs": {"config": {"n_clusters": num_clusters}},
                "generate_embeddings": {"config": {"batch_size": batch_size}}
            }
        }
        
        pipeline_run = instance.create_run_for_job(
            job_def=pipeline,
            run_config=run_config,
        )
        
        instance.launch_run(pipeline_run.run_id)
        return pipeline_run.run_id
        
    except DagsterError as e:
        logger.error(f"Dagster pipeline error: {str(e)}")
        raise RuntimeError(f"Pipeline execution failed: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error running pipeline: {str(e)}")
        raise RuntimeError(f"Unexpected error: {str(e)}")

def save_uploaded_file(uploaded_file) -> Path:
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

# Main app
try:
    st.title("Sherlog Parser")
    st.write("Upload your log file and analyze patterns using our advanced pipeline.")

    uploaded_file = st.file_uploader("Choose a log file", type=["log", "txt"])

    col1, col2 = st.columns(2)
    with col1:
        num_clusters = st.slider("Number of clusters", min_value=5, max_value=50, value=20)
    with col2:
        batch_size = st.slider("Batch size", min_value=100, max_value=5000, value=1000)

    if uploaded_file is not None:
        if st.button("Analyze Logs"):
            try:
                with st.spinner("Processing logs..."):
                    # Save the uploaded file
                    file_path = save_uploaded_file(uploaded_file)
                    
                    # Run the pipeline
                    run_id = run_pipeline(str(file_path), num_clusters, batch_size)
                    
                    st.success(f"Pipeline started! Run ID: {run_id}")
                    st.markdown(f"""
                    ### View Results
                    1. Open the [Dagster UI](http://{DAGSTER_HOST}:{DAGSTER_PORT})
                    2. Click on "Runs" in the left sidebar
                    3. Find your run with ID: `{run_id}`
                    4. View the pipeline progress and results
                    """)
            except RuntimeError as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Pipeline execution failed: {str(e)}")
            except Exception as e:
                st.error("An unexpected error occurred. Please try again.")
                logger.error(f"Unexpected error: {str(e)}")

    # Sidebar information
    st.sidebar.markdown("""
    ### Pipeline Steps
    1. **Read Log File**: Loads and validates the log file
    2. **Generate Embeddings**: Creates vector embeddings for log lines
    3. **Cluster Logs**: Groups similar log patterns
    4. **Analyze Patterns**: Extracts and validates patterns

    ### Tips
    - Adjust the number of clusters based on log variety
    - Use larger batch sizes for better performance with large files
    - Monitor progress in the Dagster UI
    """)

    # Add link to Dagster UI
    st.sidebar.markdown(f"[Open Dagster UI](http://{DAGSTER_HOST}:{DAGSTER_PORT})")

except Exception as e:
    st.error("Application failed to start. Please contact support.")
    logger.critical(f"Application error: {str(e)}", exc_info=True) 