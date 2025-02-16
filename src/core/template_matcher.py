"""Template matching and similarity comparison logic."""

from typing import List, Set, Dict, Optional
from enum import Enum
import difflib
from functools import lru_cache
from pydantic import BaseModel, Field

from src.models.ollama import VariableType

class MatchType(Enum):
    """Types of template matches."""
    EXACT = "exact"
    SIMILAR = "similar"
    VARIABLE_ONLY = "variable_only"
    NO_MATCH = "no_match"

class MatchResult(BaseModel):
    """Result of template matching."""
    match_type: MatchType = Field(description="Type of match found")
    similarity_score: float = Field(description="Similarity score between 0 and 1")
    matched_positions: List[int] = Field(description="Positions of matching tokens")
    variable_positions: List[int] = Field(description="Positions of variables in template")

class TemplateMatcher:
    """Handles template matching and similarity comparison."""
    
    def __init__(self, similarity_threshold: float = 0.8, max_examples: int = 5):
        """Initialize template matcher.
        
        Args:
            similarity_threshold: Threshold for considering templates similar
            max_examples: Maximum number of examples to keep per template
        """
        self.similarity_threshold = similarity_threshold
        self.max_examples = max_examples
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def _tokenize(template: str) -> List[str]:
        """Tokenize template into words with caching.
        
        Args:
            template: Template string to tokenize
            
        Returns:
            List of tokens
        """
        return template.split()
    
    @staticmethod
    def _get_variable_positions(tokens: List[str]) -> Set[int]:
        """Get positions of variable tokens in template.
        
        Args:
            tokens: List of template tokens
            
        Returns:
            Set of variable token positions
        """
        return {i for i, token in enumerate(tokens) 
                if any(f"<{vtype}>" in token for vtype in VariableType)}
    
    def _calculate_token_similarity(
        self,
        tokens1: List[str],
        tokens2: List[str],
        var_positions1: Set[int],
        var_positions2: Set[int]
    ) -> float:
        """Calculate similarity between token sequences.
        
        Args:
            tokens1: First token sequence
            tokens2: Second token sequence
            var_positions1: Variable positions in first sequence
            var_positions2: Variable positions in second sequence
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get static token positions
        static1 = set(range(len(tokens1))) - var_positions1
        static2 = set(range(len(tokens2))) - var_positions2
        
        # If all tokens are variables, compare variable positions
        if not static1 and not static2:
            # Normalize positions to same length
            norm_vars1 = {i/len(tokens1) for i in var_positions1}
            norm_vars2 = {i/len(tokens2) for i in var_positions2}
            intersection = len(norm_vars1 & norm_vars2)
            union = len(norm_vars1 | norm_vars2)
            return intersection / union if union > 0 else 0.0
            
        # Compare static tokens using sequence matcher
        static_tokens1 = [t for i, t in enumerate(tokens1) if i in static1]
        static_tokens2 = [t for i, t in enumerate(tokens2) if i in static2]
        
        matcher = difflib.SequenceMatcher(None, static_tokens1, static_tokens2)
        return matcher.ratio()
    
    def match(self, template1: str, template2: str) -> MatchResult:
        """Match two templates and determine their similarity.
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            MatchResult with match type and similarity details
        """
        # Tokenize templates
        tokens1 = self._tokenize(template1)
        tokens2 = self._tokenize(template2)
        
        # Get variable positions
        var_positions1 = self._get_variable_positions(tokens1)
        var_positions2 = self._get_variable_positions(tokens2)
        
        # Check for exact match
        if template1 == template2:
            return MatchResult(
                match_type=MatchType.EXACT,
                similarity_score=1.0,
                matched_positions=list(range(len(tokens1))),
                variable_positions=list(var_positions1)
            )
            
        # Calculate similarity
        similarity = self._calculate_token_similarity(
            tokens1, tokens2,
            var_positions1, var_positions2
        )
        
        # Determine match type
        if similarity >= self.similarity_threshold:
            match_type = MatchType.SIMILAR
        elif var_positions1 and var_positions2:
            match_type = MatchType.VARIABLE_ONLY
        else:
            match_type = MatchType.NO_MATCH
            
        # Find matching positions using difflib
        matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
        matched_positions = []
        for block in matcher.get_matching_blocks():
            matched_positions.extend(range(block.a, block.a + block.size))
            
        return MatchResult(
            match_type=match_type,
            similarity_score=similarity,
            matched_positions=matched_positions,
            variable_positions=list(var_positions1)
        )
    
    def merge_templates(self, template1: str, template2: str) -> Optional[str]:
        """Merge two similar templates if possible.
        
        Args:
            template1: First template
            template2: Second template
            
        Returns:
            Merged template if templates are similar enough, None otherwise
        """
        match_result = self.match(template1, template2)
        if match_result.match_type not in {MatchType.EXACT, MatchType.SIMILAR}:
            return None
            
        tokens1 = self._tokenize(template1)
        tokens2 = self._tokenize(template2)
        var_positions1 = self._get_variable_positions(tokens1)
        var_positions2 = self._get_variable_positions(tokens2)
        
        # Use the template with more specific variable types as base
        base_tokens = tokens1 if len(var_positions1) <= len(var_positions2) else tokens2
        other_tokens = tokens2 if base_tokens == tokens1 else tokens1
        
        # Merge tokens
        merged = []
        for i, (t1, t2) in enumerate(zip(base_tokens, other_tokens)):
            # If either is a variable, use the one from base template
            if i in var_positions1 or i in var_positions2:
                merged.append(t1)
            # For static tokens, use matching ones or base template
            else:
                merged.append(t1 if t1 == t2 else t1)
                
        return " ".join(merged) 