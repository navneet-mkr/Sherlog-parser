"""Module for managing data storage using DuckDB."""

import logging
from typing import Dict, List, Optional, Any, Generator
from contextlib import contextmanager
import duckdb
import pandas as pd
import re
from datetime import datetime
import numpy as np
from threading import Lock

from src.models import Settings, LogLine, LogBatch, ClusterInfo
from src.core.constants import DEFAULT_BATCH_SIZE
from src.core.utils import sanitize_column_name

logger = logging.getLogger(__name__)

class DatabaseError(Exception):
    """Base exception for database operations."""
    pass

class ConnectionError(DatabaseError):
    """Exception for connection-related errors."""
    pass

class QueryError(DatabaseError):
    """Exception for query-related errors."""
    pass

class DuckDBManager:
    """Handles data storage and querying using DuckDB with connection pooling."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the DuckDB connection pool.
        
        Args:
            settings: Optional Settings instance for configuration
            
        Raises:
            ConnectionError: If database connection fails
        """
        self.settings = settings or Settings()
        self._conn_lock = Lock()
        self._conn = None
        try:
            self._initialize_connection()
            self._initialize_schema()
        except Exception as e:
            raise ConnectionError(f"Failed to initialize database: {str(e)}")
    
    def _initialize_connection(self) -> None:
        """Initialize the database connection."""
        with self._conn_lock:
            if not self._conn:
                self._conn = duckdb.connect(self.settings.db_path)
    
    @contextmanager
    def get_connection(self) -> Generator[duckdb.DuckDBPyConnection, None, None]:
        """Get a database connection from the pool.
        
        Yields:
            A DuckDB connection
            
        Raises:
            ConnectionError: If connection cannot be established
        """
        try:
            self._initialize_connection()
            yield self._conn
        except Exception as e:
            raise ConnectionError(f"Failed to get database connection: {str(e)}")
    
    def _initialize_schema(self) -> None:
        """Initialize the database schema if it doesn't exist.
        
        Raises:
            QueryError: If schema initialization fails
        """
        try:
            with self.get_connection() as conn:
                # Create a table to store parsed log entries
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS logs (
                        log_id INTEGER PRIMARY KEY,
                        raw_log TEXT NOT NULL,
                        cluster_id INTEGER,
                        parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        parsed_fields JSON
                    )
                """)
                
                # Create a table to store cluster metadata
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cluster_patterns (
                        cluster_id INTEGER PRIMARY KEY,
                        regex_pattern TEXT,
                        sample_lines JSON,
                        center DOUBLE[],
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Create indices for better performance
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_logs_cluster_id ON logs(cluster_id);
                    CREATE INDEX IF NOT EXISTS idx_logs_parsed_at ON logs(parsed_at);
                    CREATE INDEX IF NOT EXISTS idx_cluster_patterns_active ON cluster_patterns(is_active);
                """)
                
                # Create a table to store pattern modifications
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_modifications (
                        id INTEGER PRIMARY KEY,
                        cluster_id INTEGER,
                        original_pattern TEXT,
                        modified_pattern TEXT,
                        sample_line TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (cluster_id) REFERENCES cluster_patterns(cluster_id)
                    )
                """)
                
        except Exception as e:
            raise QueryError(f"Failed to initialize schema: {str(e)}")
    
    def store_cluster_info(self, cluster: ClusterInfo) -> None:
        """Store cluster information and its regex pattern.
        
        Args:
            cluster: ClusterInfo object containing cluster data
            
        Raises:
            QueryError: If storage operation fails
        """
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO cluster_patterns (
                        cluster_id, regex_pattern, sample_lines, center, updated_at
                    )
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (cluster_id) DO UPDATE
                    SET regex_pattern = EXCLUDED.regex_pattern,
                        sample_lines = EXCLUDED.sample_lines,
                        center = EXCLUDED.center,
                        updated_at = CURRENT_TIMESTAMP
                """, [
                    cluster.cluster_id,
                    cluster.regex_pattern,
                    cluster.sample_lines,
                    cluster.center.tolist() if cluster.center is not None else None
                ])
                
        except Exception as e:
            raise QueryError(f"Failed to store cluster info: {str(e)}")
    
    def store_pattern_modification(
        self,
        cluster_id: int,
        original_pattern: str,
        modified_pattern: str,
        sample_line: str
    ) -> None:
        """Store a pattern modification for future analysis.
        
        Args:
            cluster_id: ID of the cluster
            original_pattern: Original regex pattern
            modified_pattern: Modified regex pattern
            sample_line: Log line that triggered the modification
            
        Raises:
            QueryError: If storage operation fails
        """
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO pattern_modifications (
                        cluster_id, original_pattern, modified_pattern, sample_line
                    )
                    VALUES (?, ?, ?, ?)
                """, [cluster_id, original_pattern, modified_pattern, sample_line])
                
        except Exception as e:
            raise QueryError(f"Failed to store pattern modification: {str(e)}")
    
    def get_all_clusters(self, active_only: bool = True) -> List[ClusterInfo]:
        """Get all cluster patterns from the database.
        
        Args:
            active_only: Whether to return only active clusters
            
        Returns:
            List of ClusterInfo objects
            
        Raises:
            QueryError: If retrieval fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        cluster_id,
                        regex_pattern,
                        sample_lines,
                        center
                    FROM cluster_patterns
                """
                
                if active_only:
                    query += " WHERE is_active = TRUE"
                    
                results = conn.execute(query).fetchall()
                
                return [
                    ClusterInfo(
                        cluster_id=row[0],
                        regex_pattern=row[1],
                        sample_lines=row[2],
                        center=np.array(row[3]) if row[3] is not None else None
                    )
                    for row in results
                ]
                
        except Exception as e:
            raise QueryError(f"Failed to retrieve clusters: {str(e)}")
    
    def store_log_batch(self, batch: LogBatch) -> None:
        """Store a batch of parsed log lines.
        
        Args:
            batch: LogBatch object containing the logs to store
            
        Raises:
            QueryError: If storage operation fails
        """
        try:
            # Convert batch to DataFrame
            df = batch.to_dataframe()
            
            with self.get_connection() as conn:
                # Store in DuckDB
                conn.execute("""
                    INSERT INTO logs (raw_log, cluster_id, parsed_at, parsed_fields)
                    SELECT 
                        raw_log,
                        cluster_id,
                        timestamp,
                        parsed_fields
                    FROM df
                """)
                
        except Exception as e:
            raise QueryError(f"Failed to store log batch: {str(e)}")
    
    def store_single_log(self, log: LogLine) -> None:
        """Store a single parsed log line.
        
        Args:
            log: LogLine object to store
            
        Raises:
            QueryError: If storage operation fails
        """
        try:
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT INTO logs (raw_log, cluster_id, parsed_at, parsed_fields)
                    VALUES (?, ?, ?, ?)
                """, [
                    log.raw_text,
                    log.cluster_id,
                    log.timestamp,
                    log.parsed_fields
                ])
                
        except Exception as e:
            raise QueryError(f"Failed to store single log: {str(e)}")
    
    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Retrieve cluster information from the database.
        
        Args:
            cluster_id: ID of the cluster to retrieve
            
        Returns:
            ClusterInfo object if found, None otherwise
            
        Raises:
            QueryError: If retrieval fails
        """
        try:
            with self.get_connection() as conn:
                result = conn.execute("""
                    SELECT 
                        cluster_id,
                        regex_pattern,
                        sample_lines,
                        center
                    FROM cluster_patterns
                    WHERE cluster_id = ?
                    AND is_active = TRUE
                """, [cluster_id]).fetchone()
                
                if result:
                    return ClusterInfo(
                        cluster_id=result[0],
                        regex_pattern=result[1],
                        sample_lines=result[2],
                        center=np.array(result[3]) if result[3] is not None else None
                    )
                return None
                
        except Exception as e:
            raise QueryError(f"Failed to retrieve cluster info: {str(e)}")
    
    def get_log_lines(
        self,
        cluster_id: Optional[int] = None,
        limit: int = DEFAULT_BATCH_SIZE,
        offset: int = 0
    ) -> List[LogLine]:
        """Retrieve log lines from the database.
        
        Args:
            cluster_id: Optional cluster ID to filter by
            limit: Maximum number of logs to retrieve
            offset: Number of logs to skip
            
        Returns:
            List of LogLine objects
            
        Raises:
            QueryError: If retrieval fails
        """
        try:
            with self.get_connection() as conn:
                query = """
                    SELECT 
                        raw_log,
                        cluster_id,
                        parsed_at,
                        parsed_fields
                    FROM logs
                """
                
                params = []
                if cluster_id is not None:
                    query += " WHERE cluster_id = ?"
                    params.append(cluster_id)
                    
                query += " ORDER BY parsed_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                results = conn.execute(query, params).fetchall()
                
                return [
                    LogLine(
                        raw_text=row[0],
                        cluster_id=row[1],
                        timestamp=row[2],
                        parsed_fields=row[3]
                    )
                    for row in results
                ]
                
        except Exception as e:
            raise QueryError(f"Failed to retrieve log lines: {str(e)}")
    
    def query(self, sql: str) -> pd.DataFrame:
        """Execute a custom SQL query.
        
        Args:
            sql: SQL query string
            
        Returns:
            pandas DataFrame with query results
            
        Raises:
            QueryError: If query execution fails
        """
        try:
            with self.get_connection() as conn:
                return conn.execute(sql).df()
                
        except Exception as e:
            raise QueryError(f"Failed to execute query: {str(e)}")
    
    def close(self) -> None:
        """Close the database connection."""
        with self._conn_lock:
            if self._conn:
                self._conn.close()
                self._conn = None 