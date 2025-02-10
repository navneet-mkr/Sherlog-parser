"""Setup script for the log parsing application."""

from setuptools import setup, find_packages

setup(
    name="log-parse-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.1.4",
        "scikit-learn>=1.3.2",
        "sentence-transformers>=2.2.2",
        "duckdb>=0.9.0",
        "pydantic>=2.5.2",
        "pydantic-settings>=2.1.0",
        "pydantic-ai==0.0.22",
        "ollama>=0.1.6",
        "pathway>=0.7.0",
        "litellm>=1.35.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "streamlit>=1.29.0",
        "click>=8.1.7",
        "python-dotenv>=1.0.0",
        "rich>=13.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.4",
            "pytest-cov>=4.1.0",
            "black>=24.1.1",
            "isort>=5.13.2",
            "mypy>=1.8.0",
            "pylint>=3.0.3",
        ],
        "docs": [
            "mkdocs>=1.5.3",
            "mkdocs-material>=9.5.3",
            "mkdocstrings>=0.24.0",
        ],
    },
    python_requires=">=3.10",
) 