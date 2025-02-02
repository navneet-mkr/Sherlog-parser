"""Module for real-time log parsing using pre-trained models and patterns."""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import re

from models import Settings, LogLine, ClusterInfo
from embeddings import EmbeddingGenerator
from regex_generation import RegexGenerator

logger = logging.getLogger(__name__)

class LogParser:
    """Real-time log parser using pre-trained embeddings and patterns."""
    
    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_manager = None,  # Type hint omitted to avoid circular import
        embedding_gen: Optional[EmbeddingGenerator] = None,
        regex_gen: Optional[RegexGenerator] = None
    ):
        """Initialize the log parser.
        
        Args:
            settings: Optional Settings instance for configuration
            db_manager: Optional DuckDBManager instance for pattern retrieval
            embedding_gen: Optional EmbeddingGenerator instance
            regex_gen: Optional RegexGenerator instance
        """
        self.settings = settings or Settings()
        self.db_manager = db_manager
        self.embedding_gen = embedding_gen or EmbeddingGenerator(self.settings)
        self.regex_gen = regex_gen or RegexGenerator(self.settings)
        
        # Cache for regex patterns and their sample lines
        self.pattern_cache: Dict[int, Tuple[str, List[str]]] = {}
        
    def parse_line(self, log_line: str) -> LogLine:
        """Parse a single log line using the most appropriate regex pattern.
        
        Args:
            log_line: Raw log line to parse
            
        Returns:
            LogLine object with parsed fields
        """
        # Create initial LogLine object
        parsed_log = LogLine(raw_text=log_line)
        
        try:
            # Generate embedding for the log line
            embedding = self.embedding_gen.generate_embeddings([log_line])[0]
            
            # Find the best matching pattern
            cluster_id, pattern = self._find_best_pattern(embedding, log_line)
            if cluster_id is not None:
                parsed_log.cluster_id = cluster_id
                
                # Apply the pattern
                if pattern:
                    try:
                        match = re.match(pattern, log_line)
                        if match:
                            parsed_log.parsed_fields = match.groupdict()
                    except re.error as e:
                        logger.warning(f"Error applying regex pattern: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error parsing log line: {str(e)}")
            
        return parsed_log
    
    def _find_best_pattern(
        self,
        embedding: np.ndarray,
        log_line: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Find the best matching regex pattern for a log line.
        
        Args:
            embedding: Embedding vector of the log line
            log_line: Raw log line text
            
        Returns:
            Tuple of (cluster_id, regex_pattern)
        """
        if not self.db_manager:
            return None, None
            
        try:
            # Get all cluster patterns from the database
            clusters = self._get_cluster_patterns()
            if not clusters:
                return None, None
            
            # Calculate cosine similarity with all cluster centers
            similarities = []
            for cluster in clusters:
                if cluster.center is not None:
                    sim = np.dot(embedding, cluster.center) / (
                        np.linalg.norm(embedding) * np.linalg.norm(cluster.center)
                    )
                    similarities.append((sim, cluster))
            
            if not similarities:
                return None, None
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            # Try patterns in order of similarity until one matches
            for sim, cluster in similarities:
                if cluster.regex_pattern:
                    try:
                        if re.match(cluster.regex_pattern, log_line):
                            return cluster.cluster_id, cluster.regex_pattern
                    except re.error:
                        continue
            
            # If no pattern matches well, try to modify the best matching pattern
            if similarities and similarities[0][0] > 0.8:  # High similarity threshold
                best_cluster = similarities[0][1]
                if best_cluster.regex_pattern:
                    pattern = self.regex_gen.verify_or_modify_pattern(
                        log_line,
                        best_cluster.sample_lines,
                        best_cluster.regex_pattern
                    )
                    if pattern:
                        # Store the modification if it's different
                        if pattern != best_cluster.regex_pattern:
                            self.db_manager.store_pattern_modification(
                                best_cluster.cluster_id,
                                best_cluster.regex_pattern,
                                pattern,
                                log_line
                            )
                        return best_cluster.cluster_id, pattern
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error finding best pattern: {str(e)}")
            return None, None
    
    def _get_cluster_patterns(self) -> List[ClusterInfo]:
        """Get all cluster patterns from the database or cache.
        
        Returns:
            List of ClusterInfo objects
        """
        # TODO: Implement caching with TTL
        if self.db_manager:
            return self.db_manager.get_all_clusters()
        return [] 