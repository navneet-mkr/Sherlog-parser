from setuptools import find_packages, setup

setup(
    name="log-parse-ai",
    version="0.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "sentence-transformers>=2.2.2",
        "pydantic>=2.5.2",
        "streamlit>=1.29.0",
        "dagster>=1.5.13",
        "dagster-cloud>=1.5.13",
        "huggingface-hub>=0.19.4",
        "torch>=2.1.2",
        "requests>=2.31.0",
        "tqdm>=4.66.1",
        "python-dotenv>=1.0.0",
        "httpx>=0.24.1",  # For Ollama HTTP API communication
        "diskcache>=5.6.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "mypy>=1.5.1",
            "ruff>=0.0.287",
        ]
    },
    python_requires=">=3.10",
) 