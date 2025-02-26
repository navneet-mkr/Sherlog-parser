"""Module providing LogAI-based log analysis tools."""

import logging
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta
import os

import pandas as pd
import numpy as np
from rich.console import Console
import plotly.graph_objects as go

from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig
from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.algorithms.parsing_algo.drain import DrainParams
from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor
from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
from logai.analysis.clustering import ClusteringConfig, Clustering
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.utils import constants

logger = logging.getLogger(__name__)
console = Console()

class LogAIAnalyzer:
    """Class providing LogAI-based analysis tools."""
    
    def __init__(self, log_file: str):
        """Initialize the analyzer.
        
        Args:
            log_file: Path to the log file to analyze
        """
        self.log_file = log_file
        self._setup_base_components()
        
    def _setup_base_components(self):
        """Set up basic LogAI components used by multiple tools."""
        # Data loader
        self.data_loader = OpenSetDataLoader(
            OpenSetDataLoaderConfig(
                dataset_name="system_logs",
                filepath=self.log_file,
                reader_args={
                    "log_format": "<Timestamp> <Level> <Component> <PID> <Content>"
                }
            )
        )
        
        # Load data
        self.logrecord = self.data_loader.load_data()
        
        # Preprocessor with common regex patterns
        self.preprocessor = Preprocessor(
            PreprocessorConfig(
                custom_replace_list=[
                    [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
                    [r"(?<=pid=)\d+", "<PID>"],
                    [r"(0x)[0-9a-zA-Z]+", "<HEX>"],
                    [r"\d+", "<NUM>"]
                ]
            )
        )
        
        # Parser
        self.parser = LogParser(
            LogParserConfig(
                parsing_algorithm="drain",
                parsing_algo_params=DrainParams(
                    sim_th=0.5,
                    depth=5
                )
            )
        )
        
    def detect_timeseries_anomalies(
        self,
        time_window: str = "15min",
        algorithm: str = "dbl",
        group_by: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect anomalies using time-series analysis.
        
        Args:
            time_window: Time window for aggregation
            algorithm: Anomaly detection algorithm ('dbl', 'isolation_forest', etc.)
            group_by: Optional list of fields to group by
            
        Returns:
            Tuple of (anomalies DataFrame, detection metadata)
        """
        # Clean and parse logs
        clean_logs, _ = self.preprocessor.clean_log(
            self.logrecord.body[constants.LOGLINE_NAME]
        )
        parsed_result = self.parser.parse(clean_logs)
        parsed_loglines = parsed_result['parsed_logline']
        
        # Extract time-series features
        if group_by is None:
            group_by = ['parsed_logline', 'Level', 'Component']
            
        feature_extractor = FeatureExtractor(
            FeatureExtractorConfig(
                group_by_time=time_window,
                group_by_category=group_by
            )
        )
        
        counter_vector = feature_extractor.convert_to_counter_vector(
            log_pattern=parsed_loglines,
            attributes=self.logrecord.attributes,
            timestamps=self.logrecord.timestamp['timestamp']
        )
        
        # Configure anomaly detector
        detector = AnomalyDetector(
            AnomalyDetectionConfig(
                algo_name=algorithm
            )
        )
        
        # Split data and detect anomalies
        train_size = int(len(counter_vector) * 0.7)
        train_data = counter_vector.iloc[:train_size]
        test_data = counter_vector.iloc[train_size:]
        
        detector.fit(train_data[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]])
        anomaly_scores = detector.predict(test_data[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]])
        
        # Prepare results
        anomalies = counter_vector.iloc[anomaly_scores[anomaly_scores > 0].index]
        metadata = {
            'total_windows': len(counter_vector),
            'anomaly_windows': len(anomalies),
            'algorithm': algorithm,
            'time_window': time_window
        }
        
        return anomalies, metadata
        
    def cluster_logs(
        self,
        n_clusters: int = 10,
        use_attributes: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cluster logs based on semantic similarity.
        
        Args:
            n_clusters: Number of clusters to form
            use_attributes: Whether to include log attributes in clustering
            
        Returns:
            Tuple of (clustered DataFrame, clustering metadata)
        """
        # Clean and parse logs
        clean_logs, _ = self.preprocessor.clean_log(
            self.logrecord.body[constants.LOGLINE_NAME]
        )
        parsed_result = self.parser.parse(clean_logs)
        parsed_loglines = parsed_result['parsed_logline']
        
        # Vectorize log messages
        vectorizer = LogVectorizer(
            VectorizerConfig(algo_name="word2vec")
        )
        vectorizer.fit(parsed_loglines)
        log_vectors = vectorizer.transform(parsed_loglines)
        
        # Encode attributes if requested
        if use_attributes:
            encoder = CategoricalEncoder(
                CategoricalEncoderConfig(name="label_encoder")
            )
            attributes_encoded = encoder.fit_transform(self.logrecord.attributes)
        else:
            attributes_encoded = None
            
        # Extract features
        feature_extractor = FeatureExtractor(
            FeatureExtractorConfig(max_feature_len=100)
        )
        _, feature_vector = feature_extractor.convert_to_feature_vector(
            log_vectors,
            attributes_encoded,
            self.logrecord.timestamp['timestamp']
        )
        
        # Perform clustering
        clustering = Clustering(
            ClusteringConfig(
                algo_name='kmeans',
                algo_params=KMeansParams(
                    n_clusters=n_clusters
                )
            )
        )
        
        clustering.fit(feature_vector)
        cluster_labels = clustering.predict(feature_vector)
        
        # Add cluster labels to DataFrame
        result_df = self.logrecord.to_dataframe()
        result_df['cluster'] = cluster_labels
        
        # Calculate cluster statistics
        cluster_stats = {
            'cluster_sizes': result_df['cluster'].value_counts().to_dict(),
            'silhouette_score': clustering.evaluate(feature_vector)['silhouette_score']
        }
        
        return result_df, cluster_stats
        
    def detect_semantic_anomalies(
        self,
        algorithm: str = "isolation_forest",
        train_size: float = 0.7
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect anomalies using semantic analysis.
        
        Args:
            algorithm: Anomaly detection algorithm name
            train_size: Fraction of data to use for training
            
        Returns:
            Tuple of (anomalies DataFrame, detection metadata)
        """
        # Clean and parse logs
        clean_logs, _ = self.preprocessor.clean_log(
            self.logrecord.body[constants.LOGLINE_NAME]
        )
        parsed_result = self.parser.parse(clean_logs)
        parsed_loglines = parsed_result['parsed_logline']
        
        # Vectorize logs
        vectorizer = LogVectorizer(
            VectorizerConfig(algo_name="word2vec")
        )
        vectorizer.fit(parsed_loglines)
        log_vectors = vectorizer.transform(parsed_loglines)
        
        # Encode attributes
        encoder = CategoricalEncoder(
            CategoricalEncoderConfig(name="label_encoder")
        )
        attributes_encoded = encoder.fit_transform(self.logrecord.attributes)
        
        # Extract features
        feature_extractor = FeatureExtractor(
            FeatureExtractorConfig(max_feature_len=100)
        )
        _, feature_vector = feature_extractor.convert_to_feature_vector(
            log_vectors,
            attributes_encoded,
            self.logrecord.timestamp['timestamp']
        )
        
        # Split data
        train_idx = int(len(feature_vector) * train_size)
        train_data = feature_vector[:train_idx]
        test_data = feature_vector[train_idx:]
        
        # Configure detector
        detector = AnomalyDetector(
            AnomalyDetectionConfig(
                algo_name=algorithm
            )
        )
        
        # Train and detect
        detector.fit(train_data)
        anomaly_scores = detector.predict(test_data)
        
        # Get anomalous logs
        anomaly_indices = np.where(anomaly_scores == 1)[0] + train_idx
        anomalies_df = self.logrecord.to_dataframe().iloc[anomaly_indices]
        
        metadata = {
            'total_logs': len(self.logrecord.to_dataframe()),
            'anomaly_count': len(anomalies_df),
            'algorithm': algorithm,
            'anomaly_rate': len(anomalies_df) / len(self.logrecord.to_dataframe())
        }
        
        return anomalies_df, metadata
        
    def visualize_clusters(
        self,
        clustered_df: pd.DataFrame,
        output_dir: str = "output"
    ) -> str:
        """Create visualization for clustered logs.
        
        Args:
            clustered_df: DataFrame with cluster assignments
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create timeline visualization
        fig = go.Figure()
        
        # Add trace for each cluster
        for cluster in sorted(clustered_df['cluster'].unique()):
            cluster_data = clustered_df[clustered_df['cluster'] == cluster]
            
            fig.add_trace(go.Scatter(
                x=cluster_data['timestamp'],
                y=[cluster] * len(cluster_data),
                mode='markers',
                name=f'Cluster {cluster}',
                text=cluster_data['Content'],
                hovertemplate="<b>Time:</b> %{x}<br>" +
                            "<b>Message:</b> %{text}<br>" +
                            "<b>Cluster:</b> " + str(cluster)
            ))
            
        fig.update_layout(
            title="Log Clusters Timeline",
            xaxis_title="Time",
            yaxis_title="Cluster",
            showlegend=True
        )
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/clusters_{timestamp}.html"
        fig.write_html(output_path)
        
        return output_path
        
    def visualize_anomalies(
        self,
        anomalies_df: pd.DataFrame,
        scores: Optional[pd.Series] = None,
        output_dir: str = "output"
    ) -> str:
        """Create visualization for anomalous logs.
        
        Args:
            anomalies_df: DataFrame containing anomalous logs
            scores: Optional anomaly scores
            output_dir: Directory to save visualization
            
        Returns:
            Path to saved visualization file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig = go.Figure()
        
        # Add anomaly points
        fig.add_trace(go.Scatter(
            x=anomalies_df['timestamp'],
            y=scores if scores is not None else [1] * len(anomalies_df),
            mode='markers',
            name='Anomalies',
            marker=dict(
                color='red',
                size=10
            ),
            text=anomalies_df['Content'],
            hovertemplate="<b>Time:</b> %{x}<br>" +
                        "<b>Message:</b> %{text}<br>" +
                        "<b>Score:</b> %{y:.2f}"
        ))
        
        fig.update_layout(
            title="Log Anomalies Timeline",
            xaxis_title="Time",
            yaxis_title="Anomaly Score" if scores is not None else "Anomaly",
            showlegend=True
        )
        
        # Save visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/anomalies_{timestamp}.html"
        fig.write_html(output_path)
        
        return output_path 