"""Tests for the clustering module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.core.clustering import LogClusterer
from src.models import LogLine, Settings

@pytest.fixture
def sample_embeddings():
    """Generate sample embeddings for testing."""
    rng = np.random.RandomState(42)
    return rng.randn(100, 10)

@pytest.fixture
def sample_log_lines():
    """Generate sample log lines for testing."""
    return [
        LogLine(raw_text=f"Sample log line {i}", timestamp=datetime.now())
        for i in range(100)
    ]

@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model persistence."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

def test_clusterer_initialization():
    """Test clusterer initialization with default parameters."""
    clusterer = LogClusterer()
    assert clusterer.n_clusters == 20
    assert not clusterer.is_fitted
    assert isinstance(clusterer.clusters, dict)

def test_partial_fit(sample_embeddings, sample_log_lines):
    """Test partial fit with embeddings and log lines."""
    clusterer = LogClusterer(n_clusters=5)
    clusterer.partial_fit(sample_embeddings, sample_log_lines)
    
    assert clusterer.is_fitted
    assert len(clusterer.clusters) > 0
    assert clusterer.clusterer.cluster_centers_.shape == (5, 10)

def test_empty_embeddings():
    """Test handling of empty embeddings."""
    clusterer = LogClusterer()
    with pytest.raises(ValueError, match="Empty embeddings array provided"):
        clusterer.partial_fit(np.array([]))

def test_predict_without_fit(sample_embeddings):
    """Test prediction without fitting."""
    clusterer = LogClusterer()
    with pytest.raises(ValueError, match="Clusterer must be fitted before predicting"):
        clusterer.predict(sample_embeddings)

def test_predict_after_fit(sample_embeddings):
    """Test prediction after fitting."""
    clusterer = LogClusterer(n_clusters=5)
    clusterer.partial_fit(sample_embeddings)
    
    predictions = clusterer.predict(sample_embeddings)
    assert predictions.shape == (100,)
    assert np.all(predictions >= 0) and np.all(predictions < 5)

def test_cluster_info_updates(sample_embeddings, sample_log_lines):
    """Test cluster info updates during fitting."""
    clusterer = LogClusterer(n_clusters=3)
    clusterer.partial_fit(sample_embeddings[:50], sample_log_lines[:50])
    clusterer.partial_fit(sample_embeddings[50:], sample_log_lines[50:])
    
    for cluster in clusterer.clusters.values():
        assert len(cluster.sample_lines) <= 1000  # MAX_CLUSTER_SIZE
        assert cluster.center is not None

def test_model_persistence(sample_embeddings, sample_log_lines, temp_model_dir):
    """Test model saving and loading."""
    # Train and save model
    original = LogClusterer(n_clusters=3)
    original.partial_fit(sample_embeddings, sample_log_lines)
    original.save_model(temp_model_dir)
    
    # Load model and verify
    loaded = LogClusterer.load_model(temp_model_dir)
    assert loaded.is_fitted
    assert loaded.n_clusters == original.n_clusters
    assert len(loaded.clusters) == len(original.clusters)
    
    # Verify predictions match
    np.testing.assert_array_equal(
        original.predict(sample_embeddings),
        loaded.predict(sample_embeddings)
    )

def test_cluster_statistics(sample_embeddings, sample_log_lines):
    """Test cluster statistics generation."""
    clusterer = LogClusterer(n_clusters=3)
    clusterer.partial_fit(sample_embeddings, sample_log_lines)
    
    stats = clusterer.get_cluster_statistics()
    assert len(stats) > 0
    for cluster_stats in stats.values():
        assert 'size' in cluster_stats
        assert 'has_pattern' in cluster_stats
        assert 'center_norm' in cluster_stats

def test_scaler_persistence(sample_embeddings):
    """Test that scaling is consistent after persistence."""
    clusterer = LogClusterer(n_clusters=3)
    clusterer.partial_fit(sample_embeddings)
    
    original_scaled = clusterer.scaler.transform(sample_embeddings)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        clusterer.save_model(tmpdir)
        loaded = LogClusterer.load_model(tmpdir)
        loaded_scaled = loaded.scaler.transform(sample_embeddings)
        
        np.testing.assert_array_almost_equal(original_scaled, loaded_scaled)

def test_incremental_updates(sample_embeddings, sample_log_lines):
    """Test incremental updates maintain cluster stability."""
    clusterer = LogClusterer(n_clusters=3)
    
    # First batch
    clusterer.partial_fit(sample_embeddings[:30], sample_log_lines[:30])
    centers_1 = clusterer.get_cluster_centers()
    
    # Second batch
    clusterer.partial_fit(sample_embeddings[30:60], sample_log_lines[30:60])
    centers_2 = clusterer.get_cluster_centers()
    
    # Centers should change gradually
    assert np.mean(np.abs(centers_2 - centers_1)) < 1.0

@pytest.mark.parametrize("n_clusters", [3, 5, 10])
def test_different_cluster_sizes(n_clusters, sample_embeddings):
    """Test clusterer with different numbers of clusters."""
    clusterer = LogClusterer(n_clusters=n_clusters)
    clusterer.partial_fit(sample_embeddings)
    
    predictions = clusterer.predict(sample_embeddings)
    unique_clusters = np.unique(predictions)
    assert len(unique_clusters) <= n_clusters
    assert np.all(predictions >= 0) and np.all(predictions < n_clusters) 