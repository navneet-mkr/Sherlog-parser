"""Module for log pattern extraction using grok patterns."""

import logging
from typing import Dict, List, Optional, Tuple
from pygrok import Grok
from collections import defaultdict
import re

from src.models.clustering import ClusterPattern

logger = logging.getLogger(__name__)

# Common log patterns
COMMON_PATTERNS = {
    'TIMESTAMP': r'%{YEAR}-%{MONTHNUM}-%{MONTHDAY} %{HOUR}:?%{MINUTE}(?::?%{SECOND})?',
    'LOG_LEVEL': r'(?:TRACE|DEBUG|INFO|NOTICE|WARN|ERROR|SEVERE|FATAL)',
    'THREAD': r'\[%{DATA:thread}\]',
    'CLASS': r'%{JAVACLASS:class}',
    'MESSAGE': r'%{GREEDYDATA:message}'
}

class PatternExtractor:
    """Extracts patterns from log lines using grok."""
    
    def __init__(self):
        """Initialize pattern extractor with common patterns."""
        self.patterns = {}
        
        # Initialize common patterns
        for name, pattern in COMMON_PATTERNS.items():
            self.patterns[name] = Grok(pattern)
        
        # Initialize combined patterns
        self.patterns['JAVA_LOG'] = Grok(
            r'%{TIMESTAMP:timestamp} %{LOG_LEVEL:level} %{THREAD} %{CLASS}: %{MESSAGE}'
        )
        self.patterns['SIMPLE_LOG'] = Grok(
            r'%{TIMESTAMP:timestamp} %{LOG_LEVEL:level} %{MESSAGE}'
        )
    
    def extract_pattern(self, lines: List[str], min_confidence: float = 0.8) -> Optional[ClusterPattern]:
        """Extract pattern from a list of log lines.
        
        Args:
            lines: List of log lines
            min_confidence: Minimum confidence threshold
            
        Returns:
            ClusterPattern if successful, None otherwise
        """
        if not lines:
            return None
            
        # Try common patterns first
        for pattern_name in ['JAVA_LOG', 'SIMPLE_LOG']:
            matches = [
                bool(self.patterns[pattern_name].match(line))
                for line in lines
            ]
            confidence = sum(matches) / len(matches)
            
            if confidence >= min_confidence:
                return ClusterPattern(
                    pattern=self.patterns[pattern_name].pattern,
                    confidence=confidence,
                    sample_count=len(lines)
                )
        
        # If no common pattern matches well, try to generate a custom pattern
        return self._generate_custom_pattern(lines, min_confidence)
    
    def _generate_custom_pattern(
        self,
        lines: List[str],
        min_confidence: float
    ) -> Optional[ClusterPattern]:
        """Generate a custom pattern from log lines.
        
        Args:
            lines: List of log lines
            min_confidence: Minimum confidence threshold
            
        Returns:
            ClusterPattern if successful, None otherwise
        """
        # Find common tokens
        token_positions = defaultdict(list)
        for line in lines:
            tokens = re.findall(r'\b\w+\b|\S', line)
            for i, token in enumerate(tokens):
                token_positions[token].append(i)
        
        # Identify static and variable parts
        static_tokens = {
            token: positions[0]
            for token, positions in token_positions.items()
            if len(positions) == len(lines) and len(set(positions)) == 1
        }
        
        if not static_tokens:
            return None
        
        # Build pattern
        pattern_parts = []
        current_pos = 0
        
        for token, pos in sorted(static_tokens.items(), key=lambda x: x[1]):
            if pos > current_pos:
                # Add variable part
                pattern_parts.append(r'%{DATA}')
            pattern_parts.append(re.escape(token))
            current_pos = pos + 1
        
        # Add final variable part if needed
        if current_pos < max(p[0] for p in token_positions.values()):
            pattern_parts.append(r'%{DATA}')
        
        # Create pattern
        pattern = ' '.join(pattern_parts)
        
        # Validate pattern
        grok = Grok(pattern)
        matches = [bool(grok.match(line)) for line in lines]
        confidence = sum(matches) / len(matches)
        
        if confidence >= min_confidence:
            return ClusterPattern(
                pattern=pattern,
                confidence=confidence,
                sample_count=len(lines)
            )
        
        return None
    
    def parse_line(self, line: str, pattern: str) -> Optional[Dict[str, str]]:
        """Parse a log line using a pattern.
        
        Args:
            line: Log line to parse
            pattern: Grok pattern to use
            
        Returns:
            Dictionary of parsed fields or None if no match
        """
        try:
            grok = Grok(pattern)
            match = grok.match(line)
            return match if match else None
        except Exception as e:
            logger.error(f"Error parsing line with pattern: {str(e)}")
            return None 