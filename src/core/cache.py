"""Cache implementations for log parser."""

from typing import Dict, Optional, Tuple
from datetime import datetime
import threading
from cachetools import TTLCache, Cache
from cachetools.keys import hashkey
from functools import partial

from src.models.base import CacheBase

class MemoryCache(CacheBase):
    """Thread-safe in-memory cache implementation using cachetools."""
    
    def __init__(self, ttl_seconds: int = 3600, maxsize: int = 10000):
        """Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds
            maxsize: Maximum number of items to store in cache
        """
        self._cache: Cache = TTLCache(
            maxsize=maxsize,
            ttl=ttl_seconds
        )
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        
        # Create a partial function for consistent key generation
        self._make_key = partial(hashkey)
        
    def get(self, key: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """Get cached template if exists and not expired."""
        cache_key = self._make_key(key)
        with self._lock:
            try:
                value = self._cache[cache_key]
                self._hits += 1
                return value
            except KeyError:
                self._misses += 1
                return None
            
    def put(self, key: str, value: Tuple[str, Dict[str, str]]) -> None:
        """Cache template with TTL."""
        cache_key = self._make_key(key)
        with self._lock:
            self._cache[cache_key] = value
            
    def get_statistics(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "maxsize": self._cache.maxsize,
                "currsize": len(self._cache)
            } 