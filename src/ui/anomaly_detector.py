"""Streamlit UI for real-time log anomaly detection."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.pipeline import LogProcessingPipeline
from src.core.anomaly_incidents import IncidentAnomalyDetector

def initialize_session_state():
    """Initialize session state variables."""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = LogProcessingPipeline(
            db_url=st.secrets.get('DB_URL', 'postgresql://localhost:5432/logs'),
            ollama_base_url=st.secrets.get('OLLAMA_URL', 'http://localhost:11434'),
            model_name='mistral'
        )
        st.session_state.detector = IncidentAnomalyDetector(
            pipeline=st.session_state.pipeline
        )

def render_anomaly_timeline(anomalies_df):
    """Render timeline visualization of anomalies."""
    if anomalies_df.empty:
        return
        
    fig = px.scatter(
        anomalies_df,
        x='timestamp',
        y='cluster_label',
        color='is_embedding_anomaly',
        symbol='is_numeric_anomaly',
        hover_data=['message', 'level', 'component'],
        title='Anomaly Timeline'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_cluster_distribution(anomalies_df):
    """Render cluster size distribution."""
    if anomalies_df.empty:
        return
        
    cluster_sizes = anomalies_df['cluster_label'].value_counts()
    fig = px.bar(
        x=cluster_sizes.index,
        y=cluster_sizes.values,
        title='Cluster Size Distribution',
        labels={'x': 'Cluster ID', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)

def analyze_historical_trends(
    detector: IncidentAnomalyDetector,
    table_name: str,
    current_window_hours: int
) -> Optional[Dict[str, Any]]:
    """Analyze historical trends for comparison with current anomalies.
    
    Args:
        detector: Anomaly detector instance
        table_name: Name of the logs table
        current_window_hours: Current analysis window in hours
        
    Returns:
        Dictionary containing historical metrics, or None if no historical data available
    """
    end_time = datetime.now()
    
    # Analyze last 7 days in windows of the same size as current
    windows = []
    window_end = end_time
    
    for _ in range(7 * 24 // current_window_hours):
        window_start = window_end - timedelta(hours=current_window_hours)
        
        anomalies_df = detector.detect_anomalies(
            table_name=table_name,
            hours=current_window_hours,
            additional_filters={'time_range': (window_start, window_end)}
        )
        
        if not anomalies_df.empty:
            windows.append({
                'start_time': window_start,
                'end_time': window_end,
                'total_anomalies': len(anomalies_df),
                'embedding_anomalies': anomalies_df['is_embedding_anomaly'].sum(),
                'numeric_anomalies': anomalies_df['is_numeric_anomaly'].sum(),
                'unique_components': anomalies_df['component'].nunique(),
                'error_ratio': len(anomalies_df[anomalies_df['level'].isin(['ERROR', 'CRITICAL'])]) / len(anomalies_df)
            })
            
        window_end = window_start
    
    if not windows:
        return None
        
    # Convert to DataFrame for analysis
    history_df = pd.DataFrame(windows)
    
    # Calculate historical metrics
    metrics = {
        'mean_anomalies': history_df['total_anomalies'].mean(),
        'std_anomalies': history_df['total_anomalies'].std(),
        'p95_anomalies': history_df['total_anomalies'].quantile(0.95),
        'mean_error_ratio': history_df['error_ratio'].mean(),
        'history_df': history_df
    }
    
    return metrics

def render_historical_comparison(current_anomalies_df, historical_metrics):
    """Render historical comparison visualizations."""
    if historical_metrics is None:
        st.warning("Not enough historical data for comparison")
        return
        
    history_df = historical_metrics['history_df']
    current_metrics = {
        'total_anomalies': len(current_anomalies_df),
        'error_ratio': (
            len(current_anomalies_df[current_anomalies_df['level'].isin(['ERROR', 'CRITICAL'])]) / 
            len(current_anomalies_df) if not current_anomalies_df.empty else 0
        )
    }
    
    # Create subplot with two metrics
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Anomaly Count Over Time', 'Error Ratio Over Time')
    )
    
    # Anomaly count trend
    fig.add_trace(
        go.Scatter(
            x=history_df['start_time'],
            y=history_df['total_anomalies'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # Add current window point
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[current_metrics['total_anomalies']],
            mode='markers',
            name='Current',
            marker=dict(color='red', size=10)
        ),
        row=1, col=1
    )
    
    # Error ratio trend
    fig.add_trace(
        go.Scatter(
            x=history_df['start_time'],
            y=history_df['error_ratio'],
            mode='lines+markers',
            name='Historical Error Ratio',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    # Add current error ratio point
    fig.add_trace(
        go.Scatter(
            x=[datetime.now()],
            y=[current_metrics['error_ratio']],
            mode='markers',
            name='Current Error Ratio',
            marker=dict(color='red', size=10)
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistical comparison
    st.subheader('Statistical Comparison')
    cols = st.columns(4)
    
    with cols[0]:
        deviation = (current_metrics['total_anomalies'] - historical_metrics['mean_anomalies']) / historical_metrics['std_anomalies']
        st.metric(
            'Anomaly Z-Score',
            f"{deviation:.2f}σ",
            help='Number of standard deviations from historical mean'
        )
    
    with cols[1]:
        percentile = (
            100 * (current_metrics['total_anomalies'] <= history_df['total_anomalies']).mean()
        )
        st.metric(
            'Current Percentile',
            f"{percentile:.1f}%",
            help='Percentage of historical windows with fewer anomalies'
        )
    
    with cols[2]:
        ratio = (
            current_metrics['total_anomalies'] / historical_metrics['mean_anomalies']
            if historical_metrics['mean_anomalies'] > 0 else float('inf')
        )
        st.metric(
            'Relative to Mean',
            f"{ratio:.2f}x",
            help='Ratio of current anomalies to historical mean'
        )
    
    with cols[3]:
        error_ratio_change = (
            current_metrics['error_ratio'] - historical_metrics['mean_error_ratio']
        ) * 100
        st.metric(
            'Error Ratio Change',
            f"{error_ratio_change:+.1f}%",
            help='Change in error ratio compared to historical mean'
        )

def display_summary(
    anomalies_df: pd.DataFrame,
    historical_metrics: Optional[Dict[str, Any]]
):
    """Display analysis summary in the console."""
    # ... existing code ...
    
    # Add explanation examples if available
    if 'explanation' in anomalies_df.columns:
        explained_anomalies = anomalies_df[anomalies_df['explanation'].notna()]
        if not explained_anomalies.empty:
            st.subheader('Example Anomaly Explanations')
            
            for _, row in explained_anomalies.head(5).iterrows():
                with st.expander(f"{row['level']} - {row['component'] if row['component'] else 'Unknown'}"):
                    st.text(row['message'])
                    st.markdown(f"**Explanation**: {row['explanation']}")
                    
                    details = []
                    if row['is_embedding_anomaly']:
                        if row['cluster_label'] == -1:
                            details.append("• Pattern outlier (doesn't match known clusters)")
                        else:
                            details.append(f"• Unusual cluster (cluster {row['cluster_label']})")
                    
                    if row['is_numeric_anomaly']:
                        for field, dev in zip(row['numeric_fields'], row['numeric_deviations']):
                            details.append(f"• {field}: {dev:.1f}σ deviation")
                            
                    if details:
                        st.markdown("**Technical Details**:")
                        for detail in details:
                            st.markdown(detail)

def main():
    """Main Streamlit UI."""
    st.set_page_config(
        page_title='Log Anomaly Detection',
        page_icon='🔍',
        layout='wide'
    )
    
    st.title('🔍 Real-time Log Anomaly Detection')
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar controls
    st.sidebar.header('Detection Settings')
    
    hours = st.sidebar.slider(
        'Time Window (hours)',
        min_value=1,
        max_value=24,
        value=4
    )
    
    eps = st.sidebar.slider(
        'DBSCAN Distance Threshold',
        min_value=0.1,
        max_value=1.0,
        value=0.3,
        step=0.1,
        help='Lower values create more clusters'
    )
    
    min_samples = st.sidebar.slider(
        'Minimum Cluster Size',
        min_value=2,
        max_value=10,
        value=3,
        help='Smaller values flag more anomalies'
    )
    
    log_level = st.sidebar.selectbox(
        'Log Level Filter',
        ['All'] + ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    )
    
    component = st.sidebar.text_input(
        'Component Filter',
        help='Enter component name to filter'
    )
    
    # Add explanation toggle to sidebar
    with st.sidebar.expander("Advanced Settings"):
        explain_anomalies = st.checkbox(
            'Generate Explanations',
            value=True,
            help='Use LLM to explain why logs are anomalous'
        )
        if explain_anomalies:
            max_explanations = st.number_input(
                'Max Explanations',
                min_value=1,
                max_value=1000,
                value=100,
                help='Maximum number of anomalies to explain'
            )
    
    # Update detector parameters
    st.session_state.detector.eps = eps
    st.session_state.detector.min_samples = min_samples
    st.session_state.detector.explain_anomalies = explain_anomalies
    if explain_anomalies:
        st.session_state.detector.max_explanations = max_explanations
    
    # Build filters
    filters = {}
    if log_level != 'All':
        filters['level'] = log_level
    if component:
        filters['component'] = component
    
    # Run detection on button click
    if st.sidebar.button('Detect Anomalies'):
        with st.spinner('Analyzing logs...'):
            try:
                anomalies_df = st.session_state.detector.detect_anomalies(
                    table_name='logs',  # TODO: Make configurable
                    hours=hours,
                    additional_filters=filters
                )
                
                if anomalies_df.empty:
                    st.info('No anomalies detected in the selected time window.')
                else:
                    # Display summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            'Total Anomalies',
                            len(anomalies_df)
                        )
                    with col2:
                        st.metric(
                            'Embedding Anomalies',
                            anomalies_df['is_embedding_anomaly'].sum()
                        )
                    with col3:
                        st.metric(
                            'Numeric Anomalies',
                            anomalies_df['is_numeric_anomaly'].sum()
                        )
                    
                    # Visualizations
                    st.subheader('Anomaly Timeline')
                    render_anomaly_timeline(anomalies_df)
                    
                    st.subheader('Cluster Distribution')
                    render_cluster_distribution(anomalies_df)
                    
                    # Detailed anomaly table
                    st.subheader('Detailed Anomalies')
                    st.dataframe(
                        anomalies_df[[
                            'timestamp', 'level', 'component', 'message',
                            'is_embedding_anomaly', 'is_numeric_anomaly',
                            'cluster_label'
                        ]],
                        use_container_width=True
                    )
                    
                    # Add historical analysis
                    st.subheader('Historical Trend Analysis')
                    with st.spinner('Analyzing historical trends...'):
                        historical_metrics = analyze_historical_trends(
                            st.session_state.detector,
                            'logs',
                            hours
                        )
                        render_historical_comparison(anomalies_df, historical_metrics)
                    
                    # Display summary
                    display_summary(anomalies_df, historical_metrics)
                    
                    # Export option
                    if st.download_button(
                        'Download Anomalies CSV',
                        data=anomalies_df.to_csv(index=False),
                        file_name=f'anomalies_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                        mime='text/csv'
                    ):
                        st.success('Anomalies exported successfully!')
                        
            except Exception as e:
                st.error(f'Error detecting anomalies: {str(e)}')
    
    # Auto-refresh option
    if st.sidebar.checkbox('Enable Auto-refresh'):
        refresh_interval = st.sidebar.number_input(
            'Refresh Interval (minutes)',
            min_value=1,
            max_value=60,
            value=5
        )
        st.empty()
        st.rerun()

if __name__ == '__main__':
    main() 