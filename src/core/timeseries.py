"""Module for handling time series database integration of parsed logs."""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
import json
import psycopg2
from psycopg2.extensions import connection as psycopg2_conn
from psycopg2.extras import execute_values
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import urllib.parse

logger = logging.getLogger(__name__)

class LogTimeSeriesDB:
    """Handles storing and querying parsed log data in time series format."""
    
    def __init__(self, db_url: Optional[str] = None):
        """Initialize the time series handler.
        
        Args:
            db_url: Database connection URL (e.g. postgresql://user:pass@localhost:5432/logs)
        """
        self.db_url = db_url
        self._engine: Optional[Engine] = None
        self._conn: Optional[psycopg2_conn] = None
        
    @property
    def engine(self) -> Engine:
        """Get SQLAlchemy engine for database operations."""
        if self._engine is None and self.db_url:
            self._engine = create_engine(self.db_url)
        if self._engine is None:
            raise ValueError("Database engine not initialized")
        return self._engine
    
    def connect(self) -> None:
        """Establish database connection."""
        if not self.db_url:
            raise ValueError("Database URL not provided")
        
        try:
            # Parse the URL to get connection parameters
            parsed = urllib.parse.urlparse(self.db_url)
            conn_params = {
                'dbname': parsed.path[1:],
                'user': parsed.username,
                'password': parsed.password,
                'host': parsed.hostname,
                'port': parsed.port or 5432
            }
            
            self._conn = psycopg2.connect(**conn_params)
            logger.info("Connected to TimescaleDB successfully")
            
        except Exception as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
    
    def initialize_db(self, table_name: str = 'logs') -> None:
        """Initialize database schema and TimescaleDB hypertable.
        
        Args:
            table_name: Name of the table to create
        """
        if self._conn is None:
            self.connect()
        
        if self._conn is None:
            raise ValueError("Failed to establish database connection")
            
        try:
            with self._conn.cursor() as cur:
                # Create TimescaleDB extension if not exists
                cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
                
                # Create table and hypertable
                create_table_sql = self.get_create_table_sql(table_name)
                cur.execute(create_table_sql)
                
                self._conn.commit()
                logger.info(f"Initialized TimescaleDB table: {table_name}")
                
        except Exception as e:
            if self._conn is not None:
                self._conn.rollback()
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def bulk_insert_logs(self, df: pd.DataFrame, table_name: str = 'logs', 
                        chunk_size: int = 10000) -> None:
        """Bulk insert log data into TimescaleDB.
        
        Args:
            df: DataFrame with prepared log data
            table_name: Target table name
            chunk_size: Number of rows to insert in each batch
        """
        if self._conn is None:
            self.connect()
            
        if self._conn is None:
            raise ValueError("Failed to establish database connection")
            
        try:
            # Prepare column names
            columns = list(self.get_schema().keys())
            
            # Convert DataFrame to list of tuples
            data = [tuple(x) for x in df[columns].values]
            
            with self._conn.cursor() as cur:
                # Insert data in chunks
                execute_values(
                    cur,
                    f"INSERT INTO {table_name} ({','.join(columns)}) VALUES %s",
                    data,
                    page_size=chunk_size
                )
                
            self._conn.commit()
            logger.info(f"Inserted {len(df)} rows into {table_name}")
            
        except Exception as e:
            if self._conn is not None:
                self._conn.rollback()
            logger.error(f"Error inserting data: {str(e)}")
            raise
    
    def query_logs(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """Execute a query against the log data.
        
        Args:
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise
    
    def get_example_queries(self) -> Dict[str, str]:
        """Get example TimescaleDB queries for log analysis.
        
        Returns:
            Dictionary of query names and their SQL
        """
        return {
            'error_count_by_time': """
                SELECT time_bucket('1 hour', timestamp) AS hour,
                       count(*) as error_count
                FROM logs 
                WHERE level = 'ERROR'
                GROUP BY hour 
                ORDER BY hour;
            """,
            'component_activity': """
                SELECT component,
                       time_bucket('1 hour', timestamp) AS hour,
                       count(*) as log_count
                FROM logs
                GROUP BY component, hour
                ORDER BY hour, log_count DESC;
            """,
            'template_frequency': """
                SELECT template,
                       count(*) as occurrence_count
                FROM logs
                GROUP BY template
                ORDER BY occurrence_count DESC
                LIMIT 10;
            """,
            'parameter_search': """
                SELECT timestamp, level, component, raw_message
                FROM logs
                WHERE parameters::jsonb @> '{"param_name": "param_value"}'::jsonb
                ORDER BY timestamp DESC;
            """
        }
    
    def prepare_log_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare log data for time series storage.
        
        Args:
            df: DataFrame containing raw log data
            
        Returns:
            DataFrame with structured fields ready for time series storage
        """
        parsed_df = pd.DataFrame()
        
        # Extract timestamp
        if 'Time' in df.columns:
            parsed_df['timestamp'] = pd.to_datetime(df['Time'])
        
        # Extract log level
        if 'Level' in df.columns:
            parsed_df['level'] = df['Level']
            
        # Extract component (assuming it's part of the Content field)
        def extract_component(content: str) -> str:
            # Extract component name from log content (e.g. "LabSZ sshd[24200]")
            try:
                return content.split()[1]
            except:
                return ''
                
        parsed_df['component'] = df['Content'].apply(extract_component)
        
        # Store original content
        parsed_df['raw_message'] = df['Content']
        
        # Template and parameters will be added by the log parser
        parsed_df['template'] = ''
        parsed_df['parameters'] = '{}'  # Empty JSON object
        
        return parsed_df
    
    def update_parsed_results(self, df: pd.DataFrame, templates: List[str], 
                            parameters: List[Dict]) -> pd.DataFrame:
        """Update dataframe with parsing results.
        
        Args:
            df: Prepared dataframe from prepare_log_data
            templates: List of extracted templates
            parameters: List of parameter dictionaries
            
        Returns:
            Updated DataFrame with parsing results
        """
        df = df.copy()
        df['template'] = templates
        df['parameters'] = [json.dumps(p) for p in parameters]
        return df
    
    def get_schema(self) -> Dict[str, str]:
        """Get the database schema for log storage.
        
        Returns:
            Dictionary of column names and their SQL types
        """
        return {
            'timestamp': 'TIMESTAMP NOT NULL',
            'level': 'TEXT',
            'component': 'TEXT',
            'template': 'TEXT',
            'parameters': 'JSONB',
            'raw_message': 'TEXT'
        }
    
    def get_create_table_sql(self, table_name: str = 'logs') -> str:
        """Get SQL to create the time series table.
        
        Args:
            table_name: Name of the table to create
            
        Returns:
            SQL statement to create table
        """
        schema = self.get_schema()
        columns = [f"{col} {dtype}" for col, dtype in schema.items()]
        
        return f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            {','.join(columns)}
        );
        -- Create time index
        CREATE INDEX IF NOT EXISTS idx_{table_name}_time ON {table_name} (timestamp DESC);
        -- Convert to TimescaleDB hypertable
        SELECT create_hypertable('{table_name}', 'timestamp', if_not_exists => TRUE);
        """
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Save prepared data to CSV for bulk loading.
        
        Args:
            df: DataFrame with prepared log data
            output_path: Path to save CSV file
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Saved prepared log data to {output_path}") 