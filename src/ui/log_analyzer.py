"""Streamlit interface for log analysis."""

import streamlit as st
import pandas as pd
from pathlib import Path
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.core.pipeline import LogProcessingPipeline
from src.eval.datasets import DatasetLoader

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None

def initialize_pipeline():
    """Initialize the log processing pipeline."""
    st.session_state.pipeline = LogProcessingPipeline(
        db_url=st.secrets["db_url"],
        model_name="mistral",
        cache_dir="cache"
    )

def process_uploaded_file(uploaded_file):
    """Process an uploaded log file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.log') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = Path(tmp_file.name)
    
    try:
        # Read logs
        logs_df = pd.read_csv(tmp_path, names=['Content'])
        
        # Process logs
        pipeline = st.session_state.pipeline
        table_name = f"uploaded_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process in batches
        total_logs = len(logs_df)
        processed = 0
        
        for i in range(0, total_logs, pipeline.batch_size):
            batch_df = logs_df.iloc[i:i + pipeline.batch_size]
            pipeline._process_batch(batch_df, table_name)
            
            # Update progress
            processed += len(batch_df)
            progress = processed / total_logs
            progress_bar.progress(progress)
            status_text.text(f"Processed {processed}/{total_logs} logs")
        
        return table_name
        
    finally:
        # Cleanup
        tmp_path.unlink()

def run_predefined_query(table_name, query_type):
    """Run a predefined analysis query."""
    pipeline = st.session_state.pipeline
    
    if query_type == "error_trends":
        # Error count over time
        query = f"""
        SELECT time_bucket('1 hour', timestamp) AS hour,
               level,
               count(*) as count
        FROM {table_name}
        WHERE level IN ('ERROR', 'WARN', 'INFO')
        GROUP BY hour, level
        ORDER BY hour;
        """
        df = pipeline.db.query_logs(query)
        
        # Plot
        fig = px.line(df, x='hour', y='count', color='level',
                     title='Log Levels Over Time')
        st.plotly_chart(fig)
        
    elif query_type == "component_activity":
        # Component activity
        query = f"""
        SELECT component,
               count(*) as count
        FROM {table_name}
        GROUP BY component
        ORDER BY count DESC
        LIMIT 10;
        """
        df = pipeline.db.query_logs(query)
        
        # Plot
        fig = px.bar(df, x='component', y='count',
                    title='Most Active Components')
        st.plotly_chart(fig)
        
    elif query_type == "template_patterns":
        # Template patterns
        query = f"""
        SELECT template,
               count(*) as count
        FROM {table_name}
        GROUP BY template
        ORDER BY count DESC
        LIMIT 10;
        """
        df = pipeline.db.query_logs(query)
        st.write("Most Common Log Patterns:")
        st.dataframe(df)

def main():
    """Main Streamlit interface."""
    st.title("Log Analysis Dashboard")
    
    # Initialize pipeline if needed
    if st.session_state.pipeline is None:
        initialize_pipeline()
    
    # File upload
    st.header("Upload Logs")
    uploaded_file = st.file_uploader("Choose a log file", type=['log', 'txt', 'csv'])
    
    if uploaded_file:
        if st.button("Process Logs"):
            with st.spinner("Processing logs..."):
                table_name = process_uploaded_file(uploaded_file)
                st.session_state.current_table = table_name
                st.success(f"Logs processed and stored in table: {table_name}")
    
    # Analysis section
    st.header("Log Analysis")
    
    if 'current_table' in st.session_state:
        table_name = st.session_state.current_table
        
        # Predefined analyses
        analysis_type = st.selectbox(
            "Choose Analysis",
            ["error_trends", "component_activity", "template_patterns"]
        )
        
        if st.button("Run Analysis"):
            run_predefined_query(table_name, analysis_type)
        
        # Custom query
        st.subheader("Custom Query")
        custom_query = st.text_area("Enter SQL Query:", height=100)
        
        if st.button("Run Query"):
            try:
                results = st.session_state.pipeline.db.query_logs(custom_query)
                st.dataframe(results)
            except Exception as e:
                st.error(f"Query error: {str(e)}")
        
        # Query examples
        with st.expander("Query Examples"):
            st.code("""
            -- Error logs in last hour
            SELECT timestamp, level, component, raw_message
            FROM {table_name}
            WHERE level = 'ERROR'
            AND timestamp >= NOW() - INTERVAL '1 hour'
            ORDER BY timestamp DESC;
            
            -- Component activity over time
            SELECT time_bucket('15 minutes', timestamp) AS time,
                   component,
                   count(*) as log_count
            FROM {table_name}
            GROUP BY time, component
            ORDER BY time DESC;
            
            -- Search by template pattern
            SELECT *
            FROM {table_name}
            WHERE template LIKE '%authentication%'
            ORDER BY timestamp DESC
            LIMIT 100;
            """)

if __name__ == "__main__":
    main() 