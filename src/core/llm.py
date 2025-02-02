"""Module providing a unified interface for different LLM providers using langchain."""

import logging
from typing import Any, Dict, Optional
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_community.llms import LlamaCpp, CTransformers

from src.models import Settings

logger = logging.getLogger(__name__)

class LLMManager:
    """Manager for different LLM providers using langchain."""
    
    SUPPORTED_PROVIDERS = {
        "openai": ChatOpenAI,
        "local-llama-cpp": LlamaCpp,
        "local-ctransformers": CTransformers
    }
    
    def __init__(self, settings: Settings):
        """Initialize the LLM manager.
        
        Args:
            settings: Settings instance with LLM configuration
        """
        self.settings = settings
        self.model = self._initialize_model()
    
    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get model initialization kwargs based on provider type.
        
        Returns:
            Dictionary of initialization parameters
        """
        base_kwargs = {
            "temperature": self.settings.llm.temperature
        }
        
        provider_specific_kwargs = {
            "openai": {
                "api_key": self.settings.llm.openai_api_key,
                "model_name": self.settings.llm.model_name
            },
            "local-llama-cpp": {
                "model_path": self.settings.llm.model_path,
                "max_tokens": self.settings.llm.max_tokens,
                "n_ctx": self.settings.llm.context_length,
                "verbose": True
            },
            "local-ctransformers": {
                "model": self.settings.llm.model_path,
                "model_type": "llama",
                "max_new_tokens": self.settings.llm.max_tokens,
                "context_length": self.settings.llm.context_length
            }
        }
        
        kwargs = base_kwargs.copy()
        kwargs.update(provider_specific_kwargs.get(self.settings.llm.provider_type, {}))
        return kwargs
    
    def _initialize_model(self) -> BaseChatModel:
        """Initialize the appropriate LLM model based on settings.
        
        Returns:
            Initialized langchain model
            
        Raises:
            ValueError: If provider type is not supported
        """
        provider_class = self.SUPPORTED_PROVIDERS.get(self.settings.llm.provider_type)
        if not provider_class:
            raise ValueError(
                f"Unsupported provider type: {self.settings.llm.provider_type}. "
                f"Supported types: {list(self.SUPPORTED_PROVIDERS.keys())}"
            )
            
        return provider_class(**self._get_model_kwargs())
    
    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate a completion for the given prompt.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override
            
        Returns:
            Generated completion text
        """
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        if temperature is not None:
            # Create a new model instance with the specified temperature
            model = self._initialize_model()
            model.temperature = temperature
        else:
            model = self.model
            
        response = model.generate([messages])
        return response.generations[0][0].text
    
    def generate_structured_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        response_model: Any = None
    ) -> Any:
        """Generate a structured completion using a Pydantic model.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override
            response_model: Optional Pydantic model for response validation
            
        Returns:
            Validated response object or raw completion if no model provided
        """
        completion = self.generate_completion(prompt, system_prompt, temperature)
        
        if not response_model:
            return completion
            
        try:
            # Try to extract JSON from the completion
            json_start = completion.find("{")
            json_end = completion.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = completion[json_start:json_end]
                return response_model.model_validate_json(json_str)
        except Exception as e:
            logger.error(f"Error parsing structured response: {str(e)}")
            return None

    @staticmethod
    def list_available_models() -> Dict[str, list]:
        """List available local models in the models directory.
        
        Returns:
            Dictionary with model types and their available models
        """
        models_dir = Path("models")
        if not models_dir.exists():
            return {"gguf": [], "ggml": [], "other": []}
            
        models = {
            "gguf": [],
            "ggml": [],
            "other": []
        }
        
        for model_file in models_dir.glob("*"):
            if model_file.is_file():
                model_info = {
                    "name": model_file.stem,
                    "path": str(model_file),
                    "size": model_file.stat().st_size // (1024 * 1024)  # Size in MB
                }
                
                if model_file.suffix == ".gguf":
                    models["gguf"].append(model_info)
                elif model_file.suffix == ".ggml":
                    models["ggml"].append(model_info)
                else:
                    models["other"].append(model_info)
                    
        return models 