import os
import logging
import streamlit as st
from pathlib import Path
from dagster import DagsterInstance

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize constants
DAGSTER_HOST = os.getenv("DAGSTER_HOST", "localhost")
DAGSTER_PORT = os.getenv("DAGSTER_PORT", "3000")
DAGSTER_GRPC_HOST = os.getenv("DAGSTER_GRPC_HOST", "localhost")
DAGSTER_GRPC_PORT = os.getenv("DAGSTER_GRPC_PORT", "4000")
DATA_DIR = Path("/data/logs")

def init_dagster():
    """Initialize Dagster instance.
    
    Returns:
        DagsterInstance: The Dagster instance
        
    Raises:
        RuntimeError: If Dagster initialization fails
    """
    try:
        instance = DagsterInstance.get()
        return instance
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
        instance = init_dagster()
        
        # Configure and launch the job
        run_config = {
            "ops": {
                "read_log_file": {
                    "config": {
                        "file_path": str(file_path),
                        "encoding": "utf-8"
                    }
                },
                "cluster_logs": {
                    "config": {"n_clusters": num_clusters}
                },
                "generate_embeddings": {
                    "config": {"batch_size": batch_size}
                }
            }
        }
        
        # Create and launch the run
        pipeline_run = instance.create_run_for_job(
            job_name="log_processing_job",
            run_config=run_config,
        )
        
        instance.launch_run(pipeline_run.run_id)
        return pipeline_run.run_id
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise RuntimeError(f"Pipeline execution failed: {str(e)}")

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
                    
                    # Run the pipeline
                    run_id = run_pipeline(str(file_path), num_clusters, batch_size)
                    
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