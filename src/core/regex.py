"""Module for generating regex patterns using Ollama."""

import logging
import re
from typing import List, Optional
import json
from pathlib import Path
import httpx

from src.models import (
    RegexGenerationPrompt,
    RegexGenerationResponse,
    Settings
)
from src.core.constants import (
    REGEX_SYSTEM_PROMPT,
    PATTERN_CHOICE_SYSTEM_PROMPT,
    MAX_SAMPLE_LINES,
    MAX_RETRIES,
    DEFAULT_TEMPERATURE,
    RETRY_TEMPERATURE
)

logger = logging.getLogger(__name__)

class RegexGenerator:
    """Handles generation of regex patterns using Ollama."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the regex generator.
        
        Args:
            settings: Optional Settings instance for configuration
        """
        self.settings = settings or Settings()
        self.base_url = f"http://{self.settings.ollama.host}:{self.settings.ollama.port}"
        
    async def _generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE
    ) -> str:
        """Generate a completion using Ollama.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt
            temperature: Temperature for generation
            
        Returns:
            Generated text
            
        Raises:
            httpx.HTTPError: If the API request fails
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.settings.ollama.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": self.settings.ollama.num_predict,
                        "top_k": self.settings.ollama.top_k,
                        "top_p": self.settings.ollama.top_p
                    }
                }
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
    
    async def generate_regex_for_cluster(
        self,
        sample_lines: List[str],
        max_retries: int = MAX_RETRIES,
        field_requirements: Optional[List[str]] = None
    ) -> Optional[str]:
        """Generate a regex pattern that matches the sample log lines.
        
        Args:
            sample_lines: List of sample log lines from a cluster
            max_retries: Maximum number of attempts to generate a valid regex
            field_requirements: Optional list of required field names
            
        Returns:
            A regex pattern string if successful, None otherwise
            
        Raises:
            ValueError: If no sample lines are provided
        """
        if not sample_lines:
            raise ValueError("At least one sample line is required")
            
        # Create prompt using Pydantic model
        prompt = RegexGenerationPrompt(
            sample_lines=sample_lines[:MAX_SAMPLE_LINES],
            field_requirements=field_requirements or RegexGenerationPrompt.__fields__['field_requirements'].default
        )
        
        for attempt in range(max_retries):
            try:
                # Generate pattern using Ollama
                completion = await self._generate_completion(
                    prompt=prompt.format_prompt(),
                    system_prompt=REGEX_SYSTEM_PROMPT,
                    temperature=DEFAULT_TEMPERATURE if attempt == 0 else RETRY_TEMPERATURE
                )
                
                try:
                    # Try to parse the response as JSON
                    result = RegexGenerationResponse.model_validate_json(completion)
                    if result and self._validate_pattern(result.pattern, sample_lines):
                        logger.info(f"Successfully generated pattern with fields: {list(result.field_descriptions.keys())}")
                        return result.pattern
                except Exception as e:
                    logger.error(f"Error parsing response: {str(e)}")
                    continue
                    
            except Exception as e:
                logger.error(f"Error generating regex (attempt {attempt + 1}): {str(e)}")
                continue
                
        logger.error("Failed to generate valid regex pattern after all attempts")
        return None
    
    def _validate_pattern(self, pattern: str, sample_lines: List[str]) -> bool:
        """Validate that a regex pattern matches all sample lines.
        
        Args:
            pattern: Regex pattern to validate
            sample_lines: List of sample lines to test against
            
        Returns:
            True if pattern matches all lines and captures data, False otherwise
            
        Raises:
            re.error: If the pattern is invalid
        """
        try:
            regex = re.compile(pattern)
            matches = [regex.match(line) for line in sample_lines]
            
            # Check if all lines match and all have at least one group
            if not all(matches):
                logger.debug("Pattern does not match all lines")
                return False
                
            # Verify that each match captures meaningful data
            for i, match in enumerate(matches):
                if not match or not match.groupdict():
                    logger.debug(f"No named groups captured for line {i + 1}")
                    return False
                    
            return True
            
        except re.error as e:
            logger.warning(f"Invalid regex pattern: {str(e)}")
            raise
    
    async def verify_or_modify_pattern(
        self,
        log_line: str,
        sample_lines: List[str],
        existing_pattern: str
    ) -> Optional[str]:
        """Ask Ollama to verify or modify a pattern for a new log line.
        
        Args:
            log_line: Current log line to parse
            sample_lines: Sample lines from the cluster
            existing_pattern: Existing regex pattern
            
        Returns:
            Verified or modified regex pattern, None if rejected
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not log_line or not existing_pattern:
            raise ValueError("Log line and existing pattern are required")
            
        prompt = f"""Analyze this new log line:
{log_line}

Compare it to these similar logs:
{chr(10).join(sample_lines[:MAX_SAMPLE_LINES])}

Current regex pattern:
{existing_pattern}

Should this pattern be:
1. Used as-is (if it matches the new line)
2. Modified (if it needs small changes)
3. Rejected (if the line is too different)

Respond with ONLY:
KEEP: {existing_pattern}
or
MODIFY: <new_pattern>
or
REJECT"""
        
        try:
            result = await self._generate_completion(
                prompt=prompt,
                system_prompt=PATTERN_CHOICE_SYSTEM_PROMPT,
                temperature=DEFAULT_TEMPERATURE
            )
            
            result = result.strip()
            if result.startswith("KEEP:"):
                return existing_pattern
            elif result.startswith("MODIFY:"):
                new_pattern = result.split(":", 1)[1].strip()
                if self._validate_pattern(new_pattern, [log_line, *sample_lines[:MAX_SAMPLE_LINES]]):
                    return new_pattern
                return existing_pattern
                    
            return None
            
        except Exception as e:
            logger.error(f"Error verifying pattern: {str(e)}")
            return None 