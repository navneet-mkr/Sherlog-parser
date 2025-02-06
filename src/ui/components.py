import streamlit as st
from typing import Dict, Any, Optional, List
from src.utils.progress import format_size, ProgressStats

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

def show_file_details(files: List[Any]):
    """Show details of uploaded files."""
    st.markdown("### 📋 Selected Files")
    for file in files:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{file.name}**")
        with col2:
            st.markdown(f"Size: {format_size(file.size)}")

def show_model_info(model_info: Dict[str, Any]):
    """Display model information."""
    st.markdown(f"""
    **Model Information**:
    - Size: {format_size(model_info.get('size', 0))}
    - Format: {model_info.get('details', {}).get('format', 'Unknown')}
    - Family: {model_info.get('details', {}).get('family', 'Unknown')}
    - Parameters: {model_info.get('details', {}).get('parameter_size', 'Unknown')}
    - Quantization: {model_info.get('details', {}).get('quantization_level', 'Unknown')}
    """)

def show_progress(stats: ProgressStats, container=st):
    """Show download/processing progress."""
    progress_html = f"""
    <div style="margin: 10px 0; font-size: 0.9em;">
        <div style="color: #666;">
            <span style="display: inline-block; width: 80px;">Size:</span>
            {format_size(stats.completed)} / {format_size(stats.total)} ({stats.progress_pct:.1f}%)
        </div>
        <div style="color: #666;">
            <span style="display: inline-block; width: 80px;">Speed:</span>
            {format_size(stats.speed)}/s
        </div>
        <div style="color: #666;">
            <span style="display: inline-block; width: 80px;">ETA:</span>
            {stats.eta}
        </div>
    </div>
    """
    container.markdown(progress_html, unsafe_allow_html=True)

def show_sidebar_info(dagster_host: str, dagster_port: str):
    """Show sidebar information."""
    st.sidebar.header("📖 Pipeline Steps")
    st.sidebar.markdown("""
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

    st.sidebar.header("💡 Tips")
    st.sidebar.markdown("""
    - Adjust clusters based on log variety
    - Use larger batch sizes for better performance
    - Monitor progress in the Dagster UI
    """)

    # Add link to Dagster UI
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"[🔗 Open Dagster UI](http://{dagster_host}:{dagster_port})")

def show_model_settings() -> Dict[str, float]:
    """Show and get model settings."""
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
        
        return {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        } 