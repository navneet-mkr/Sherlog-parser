"""Constants used throughout the core modules."""

# LLM System Prompts
REGEX_SYSTEM_PROMPT = """You are an expert regex pattern generator.
Your task is to analyze log lines and create precise, reliable Python regex patterns.
Focus on accuracy and robustness of the patterns.
Use clear, descriptive names for capture groups."""

PATTERN_CHOICE_SYSTEM_PROMPT = """You are an expert at analyzing log patterns.
Your task is to determine if a regex pattern needs modification to match a new log line.
If the existing pattern is suitable, return it unchanged.
If it needs modification, suggest the minimal change needed."""

SQL_GENERATION_PROMPT = """You are an expert SQL query generator.
Your task is to convert natural language questions into DuckDB SQL queries.
Focus on writing efficient, accurate queries that answer the user's question."""

# Database Configuration
DEFAULT_BATCH_SIZE = 1000
MAX_SAMPLE_LINES = 5
MAX_RETRIES = 3

# File Processing
CHUNK_SIZE = 10000
MAX_FILE_SIZE = 1024 * 1024 * 100  # 100MB

# Model Configuration
DEFAULT_TEMPERATURE = 0.1
RETRY_TEMPERATURE = 0.2
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Storage
DEFAULT_DB_PATH = "logs.duckdb"
SUPPORTED_FILE_TYPES = {".log", ".txt", ".json"}

# Clustering
DEFAULT_N_CLUSTERS = 20
MIN_CLUSTER_SIZE = 10
MAX_CLUSTER_SIZE = 1000

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = 'INFO' 