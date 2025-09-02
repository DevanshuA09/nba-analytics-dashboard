"""
Redis Caching Service for NBA Analytics Dashboard
Provides caching for NBA API queries and model inference
"""

import redis
import json
import pickle
import logging
from typing import Any, Optional, Union
from datetime import timedelta
import os

logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, decode_responses: bool = True):
        """
        Initialize Redis cache service
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            decode_responses: Whether to decode responses
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.decode_responses = decode_responses
        
        # Try to connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=decode_responses,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.redis_client.ping()
            self.available = True
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Using in-memory cache fallback.")
            self.available = False
            self._memory_cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            if self.available:
                value = self.redis_client.get(key)
                if value:
                    # Try to deserialize JSON first, then pickle
                    try:
                        return json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        try:
                            return pickle.loads(value)
                        except:
                            return value
                return None
            else:
                # Fallback to memory cache
                return self._memory_cache.get(key)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """
        Set value in cache with TTL
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.available:
                # Try to serialize as JSON first, then pickle
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    serialized_value = pickle.dumps(value)
                
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                # Fallback to memory cache
                self._memory_cache[key] = value
                return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.available:
                return bool(self.redis_client.delete(key))
            else:
                # Fallback to memory cache
                if key in self._memory_cache:
                    del self._memory_cache[key]
                    return True
                return False
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache
        
        Args:
            key: Cache key
            
        Returns:
            True if key exists, False otherwise
        """
        try:
            if self.available:
                return bool(self.redis_client.exists(key))
            else:
                return key in self._memory_cache
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern
        
        Args:
            pattern: Redis key pattern (e.g., "nba_api_*")
            
        Returns:
            Number of keys deleted
        """
        try:
            if self.available:
                keys = self.redis_client.keys(pattern)
                if keys:
                    return self.redis_client.delete(*keys)
                return 0
            else:
                # Fallback to memory cache
                keys_to_delete = [k for k in self._memory_cache.keys() if pattern.replace('*', '') in k]
                for key in keys_to_delete:
                    del self._memory_cache[key]
                return len(keys_to_delete)
        except Exception as e:
            logger.error(f"Error clearing cache pattern {pattern}: {e}")
            return 0
    
    def get_ttl(self, key: str) -> int:
        """
        Get TTL for a key
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        try:
            if self.available:
                return self.redis_client.ttl(key)
            else:
                # Memory cache doesn't have TTL
                return -1 if key in self._memory_cache else -2
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -2
    
    def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment a numeric value in cache
        
        Args:
            key: Cache key
            amount: Amount to increment by
            
        Returns:
            New value after increment
        """
        try:
            if self.available:
                return self.redis_client.incrby(key, amount)
            else:
                # Fallback to memory cache
                current_value = self._memory_cache.get(key, 0)
                new_value = current_value + amount
                self._memory_cache[key] = new_value
                return new_value
        except Exception as e:
            logger.error(f"Error incrementing cache key {key}: {e}")
            return 0
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            if self.available:
                info = self.redis_client.info()
                return {
                    "connected": True,
                    "redis_version": info.get("redis_version"),
                    "used_memory": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses"),
                    "hit_rate": info.get("keyspace_hits", 0) / max(1, info.get("keyspace_hits", 0) + info.get("keyspace_misses", 0))
                }
            else:
                return {
                    "connected": False,
                    "memory_cache_size": len(self._memory_cache),
                    "fallback_mode": True
                }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}

# Cache configuration
CACHE_CONFIG = {
    "nba_api": {
        "ttl": 3600,  # 1 hour for NBA API data
        "pattern": "nba_api_*"
    },
    "model_inference": {
        "ttl": 1800,  # 30 minutes for model predictions
        "pattern": "model_*"
    },
    "chatbot": {
        "ttl": 300,   # 5 minutes for chatbot responses
        "pattern": "chat_*"
    },
    "team_stats": {
        "ttl": 7200,  # 2 hours for team statistics
        "pattern": "team_stats_*"
    },
    "player_stats": {
        "ttl": 1800,  # 30 minutes for player statistics
        "pattern": "player_stats_*"
    }
}

# Global cache instance
cache_service = None

def get_cache_service() -> CacheService:
    """Get global cache service instance"""
    global cache_service
    if cache_service is None:
        # Try to get Redis connection from environment
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        cache_service = CacheService(
            host=redis_host,
            port=redis_port,
            password=redis_password
        )
    return cache_service

def cache_key_builder(prefix: str, *args) -> str:
    """
    Build cache key from prefix and arguments
    
    Args:
        prefix: Key prefix
        *args: Arguments to include in key
        
    Returns:
        Formatted cache key
    """
    key_parts = [prefix] + [str(arg) for arg in args]
    return "_".join(key_parts)

def cached_function(cache_service: CacheService, ttl: int = 3600, key_prefix: str = "func"):
    """
    Decorator for caching function results
    
    Args:
        cache_service: Cache service instance
        ttl: Time to live in seconds
        key_prefix: Key prefix for cache
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Build cache key
            key = cache_key_builder(key_prefix, func.__name__, *args, *kwargs.items())
            
            # Try to get from cache
            cached_result = cache_service.get(key)
            if cached_result is not None:
                logger.info(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_service.set(key, result, ttl)
            logger.info(f"Cached result for {func.__name__}")
            
            return result
        return wrapper
    return decorator
