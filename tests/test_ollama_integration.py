import pytest
import requests
from unittest.mock import Mock, patch
from src.models.config import ModelInfo, LLMConfig, OllamaSettings
from src.ui.app import (
    get_available_models,
    get_llm,
    pull_model,
    delete_model,
    get_model_details
)

# Test configuration
TEST_SETTINGS = OllamaSettings(
    host="http://localhost",
    port=11434
)

@pytest.fixture
def mock_ollama_response():
    """Mock successful Ollama API response."""
    return {
        "models": [
            {"name": "llama2"},
            {"name": "mistral"},
            {"name": "codellama"}
        ]
    }

@pytest.fixture
def mock_model_details():
    """Mock model details response."""
    return {
        "license": "MIT",
        "modelfile": "FROM llama2\nPARAMETER temperature 0.7",
        "parameters": {
            "temperature": 0.7,
            "context_length": 4096
        },
        "template": "{{ .Prompt }}"
    }

def test_get_available_models(mock_ollama_response):
    """Test getting available models from Ollama."""
    with patch('requests.get') as mock_get:
        mock_get.return_value = Mock(
            status_code=200,
            json=lambda: mock_ollama_response
        )
        
        models = get_available_models()
        assert len(models) == 3
        assert "llama2" in models
        assert "mistral" in models
        assert isinstance(models["llama2"], ModelInfo)

def test_get_available_models_failure():
    """Test handling of failed model fetch."""
    with patch('requests.get') as mock_get:
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        models = get_available_models()
        # Should fall back to predefined models
        assert len(models) > 0
        assert all(isinstance(m, ModelInfo) for m in models.values())

def test_get_llm():
    """Test LLM initialization."""
    config = LLMConfig(
        model_id="llama2",
        temperature=0.7,
        num_predict=2048,
        top_k=40,
        top_p=0.9
    )
    
    with patch('langchain_community.llms.Ollama') as mock_ollama:
        mock_ollama.return_value = Mock()
        
        llm_config = get_llm(config)
        assert "llm" in llm_config
        assert "config" in llm_config
        assert llm_config["config"]["name"] == "llama2"

def test_pull_model():
    """Test model pulling."""
    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            iter_lines=lambda: [b'{"status": "downloading"}', b'{"completed": true}']
        )
        
        assert pull_model("llama2") is True

def test_pull_model_failure():
    """Test model pull failure."""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        assert pull_model("llama2") is False

def test_delete_model():
    """Test model deletion."""
    with patch('requests.delete') as mock_delete:
        mock_delete.return_value = Mock(status_code=200)
        
        assert delete_model("llama2") is True

def test_delete_model_failure():
    """Test model deletion failure."""
    with patch('requests.delete') as mock_delete:
        mock_delete.return_value = Mock(status_code=404)
        
        assert delete_model("llama2") is False

def test_get_model_details(mock_model_details):
    """Test getting model details."""
    with patch('requests.post') as mock_post:
        mock_post.return_value = Mock(
            status_code=200,
            json=lambda: mock_model_details
        )
        
        details = get_model_details("llama2")
        assert details is not None
        assert "parameters" in details
        assert "template" in details

def test_get_model_details_failure():
    """Test handling of failed model details fetch."""
    with patch('requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError()
        
        details = get_model_details("llama2")
        assert details is None

@pytest.mark.integration
def test_ollama_service_health():
    """Test Ollama service health (requires running service)."""
    try:
        response = requests.get(f"{TEST_SETTINGS.host}:{TEST_SETTINGS.port}/api/tags")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    except requests.exceptions.ConnectionError:
        pytest.fail("Ollama service is not running") 