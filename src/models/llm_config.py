from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field
import psutil
import torch

class ModelType(str, Enum):
    LOCAL = "local"
    API = "api"

class ModelProvider(str, Enum):
    LLAMA_CPP = "llama.cpp"
    OPENAI = "openai"

class ModelQuantization(str, Enum):
    Q4_K_M = "Q4_K_M"  # Good balance of quality and memory usage
    Q5_K_M = "Q5_K_M"  # Better quality, more memory
    Q8_0 = "Q8_0"      # Best quality, most memory

class ModelInfo(BaseModel):
    """Information about a specific model."""
    name: str
    size: str = "7B"
    description: str
    memory_required: int  # in MB
    avg_speed: float     # tokens/second
    quality_score: float # 1-10 scale
    quantization: ModelQuantization = ModelQuantization.Q4_K_M
    model_path: Optional[str] = None
    download_url: Optional[str] = None

class ModelConfig(BaseModel):
    """Configuration for model selection and management."""
    model_type: ModelType = ModelType.LOCAL
    provider: ModelProvider = ModelProvider.LLAMA_CPP
    api_key: Optional[str] = None
    model_path: Optional[str] = None
    quantization: ModelQuantization = ModelQuantization.Q4_K_M

    @staticmethod
    def get_available_models() -> Dict[str, ModelInfo]:
        """Get list of available models with their information."""
        return {
            "mistral-7b": ModelInfo(
                name="Mistral 7B",
                description="Fast and efficient general-purpose model. Good balance of performance and resource usage.",
                memory_required=8000,  # 8GB
                avg_speed=150.0,
                quality_score=8.0,
                download_url="https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
            ),
            "mistral-7b-instruct": ModelInfo(
                name="Mistral 7B Instruct",
                description="Instruction-tuned version of Mistral 7B. Better for following specific instructions.",
                memory_required=8000,
                avg_speed=145.0,
                quality_score=8.5,
                download_url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
            ),
            "llama2-7b": ModelInfo(
                name="Llama 2 7B",
                description="Meta's latest model. Good all-round performance.",
                memory_required=8000,
                avg_speed=140.0,
                quality_score=7.5,
                download_url="https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
            ),
            "codellama-7b": ModelInfo(
                name="CodeLlama 7B",
                description="Specialized for code understanding and generation.",
                memory_required=8000,
                avg_speed=135.0,
                quality_score=8.0,
                download_url="https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf"
            ),
            "openchat-7b": ModelInfo(
                name="OpenChat 7B",
                description="Optimized for dialogue and interaction.",
                memory_required=8000,
                avg_speed=145.0,
                quality_score=8.0,
                download_url="https://huggingface.co/TheBloke/OpenChat-3.5-GGUF/resolve/main/openchat_3.5.Q4_K_M.gguf"
            )
        }

    @staticmethod
    def get_system_memory() -> int:
        """Get available system memory in MB."""
        return psutil.virtual_memory().available // (1024 * 1024)

    @staticmethod
    def get_gpu_memory() -> Optional[int]:
        """Get available GPU memory in MB if GPU is available."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return None

    def can_run_model(self, model_info: ModelInfo) -> bool:
        """Check if the system can run a specific model."""
        system_memory = self.get_system_memory()
        gpu_memory = self.get_gpu_memory()
        
        # Add 2GB buffer for system operations
        required_memory = model_info.memory_required + 2000
        
        if gpu_memory is not None:
            return gpu_memory >= required_memory
        return system_memory >= required_memory 