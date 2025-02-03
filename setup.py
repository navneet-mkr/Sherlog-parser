from setuptools import find_packages, setup

setup(
    name="log_parse_ai",
    packages=find_packages(exclude=["tests"]),
    install_requires=[
        "dagster",
        "dagster-cloud",
        "streamlit",
        "pandas",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "diskcache",
        "pydantic",
        "langchain",
        "langchain-community",
        "huggingface-hub",
        "llama-cpp-python",
    ],
    extras_require={"dev": ["dagit", "pytest"]},
) 