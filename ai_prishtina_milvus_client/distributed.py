"""
Distributed processing and caching capabilities for Milvus operations.
"""

import os
from typing import List, Dict, Any, Optional, Union
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import redis
from functools import lru_cache
import hashlib
import json
import time
from pydantic import BaseModel, Field

from ai_prishtina_milvus_client import MilvusClient, MilvusConfig


class CacheConfig(BaseModel):
    """Configuration for caching."""
    enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    ttl: int = 3600  # Time to live in seconds
    max_size: int = 1000  # Maximum number of items in LRU cache


class DistributedConfig(BaseModel):
    """Configuration for distributed processing."""
    enabled: bool = True
    num_workers: int = Field(default=os.cpu_count() or 4)
    chunk_size: int = 1000
    use_processes: bool = True  # Use processes instead of threads
    timeout: int = 300  # Timeout in seconds


class DistributedMilvusClient(MilvusClient):
    """Milvus client with distributed processing and caching capabilities."""
    
    def __init__(
        self,
        config: MilvusConfig,
        cache_config: Optional[CacheConfig] = None,
        distributed_config: Optional[DistributedConfig] = None
    ):
        super().__init__(config)
        self.cache_config = cache_config or CacheConfig()
        self.distributed_config = distributed_config or DistributedConfig()
        
        # Initialize Redis client if caching is enabled
        if self.cache_config.enabled:
            self.redis_client = redis.from_url(self.cache_config.redis_url)
        
        # Initialize LRU cache for local caching
        self._init_lru_cache()
    
    def _init_lru_cache(self):
        """Initialize LRU cache with configured size."""
        self._lru_cache = lru_cache(maxsize=self.cache_config.max_size)
    
    def _get_cache_key(self, operation: str, params: Dict[str, Any]) -> str:
        """Generate cache key for an operation."""
        # Sort parameters to ensure consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        return f"milvus:{operation}:{hashlib.md5(sorted_params.encode()).hexdigest()}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache (Redis or LRU)."""
        if not self.cache_config.enabled:
            return None
            
        # Try Redis first
        if hasattr(self, 'redis_client'):
            cached = self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
        
        # Try LRU cache
        return self._lru_cache(cache_key)
    
    def _set_in_cache(self, cache_key: str, value: Any):
        """Set value in cache (Redis and LRU)."""
        if not self.cache_config.enabled:
            return
            
        # Set in Redis
        if hasattr(self, 'redis_client'):
            self.redis_client.setex(
                cache_key,
                self.cache_config.ttl,
                json.dumps(value)
            )
        
        # Set in LRU cache
        self._lru_cache(cache_key, value)
    
    def _process_chunk(
        self,
        chunk: List[Any],
        operation: str,
        **kwargs
    ) -> List[Any]:
        """Process a chunk of data."""
        if operation == "insert":
            return self.insert_vectors(chunk, **kwargs)
        elif operation == "search":
            return self.search_vectors(chunk, **kwargs)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _distributed_process(
        self,
        data: List[Any],
        operation: str,
        **kwargs
    ) -> List[Any]:
        """Process data in parallel using multiple workers."""
        if not self.distributed_config.enabled:
            return self._process_chunk(data, operation, **kwargs)
        
        # Split data into chunks
        chunks = [
            data[i:i + self.distributed_config.chunk_size]
            for i in range(0, len(data), self.distributed_config.chunk_size)
        ]
        
        # Choose executor based on configuration
        executor_class = ProcessPoolExecutor if self.distributed_config.use_processes else ThreadPoolExecutor
        
        # Process chunks in parallel
        with executor_class(max_workers=self.distributed_config.num_workers) as executor:
            futures = [
                executor.submit(self._process_chunk, chunk, operation, **kwargs)
                for chunk in chunks
            ]
            
            # Collect results
            results = []
            for future in futures:
                try:
                    chunk_result = future.result(timeout=self.distributed_config.timeout)
                    results.extend(chunk_result)
                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    raise
        
        return results
    
    def insert_vectors(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        partition_name: Optional[str] = None,
        **kwargs
    ) -> List[int]:
        """Insert vectors with distributed processing."""
        # Generate cache key
        cache_key = self._get_cache_key("insert", {
            "vectors": vectors,
            "metadata": metadata,
            "partition": partition_name,
            **kwargs
        })
        
        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Process in parallel
        result = self._distributed_process(
            vectors,
            "insert",
            metadata=metadata,
            partition_name=partition_name,
            **kwargs
        )
        
        # Cache result
        self._set_in_cache(cache_key, result)
        return result
    
    def search_vectors(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        partition_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """Search vectors with distributed processing and caching."""
        # Generate cache key
        cache_key = self._get_cache_key("search", {
            "query_vectors": query_vectors,
            "top_k": top_k,
            "partition_names": partition_names,
            **kwargs
        })
        
        # Check cache
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        # Process in parallel
        result = self._distributed_process(
            query_vectors,
            "search",
            top_k=top_k,
            partition_names=partition_names,
            **kwargs
        )
        
        # Cache result
        self._set_in_cache(cache_key, result)
        return result
    
    def clear_cache(self):
        """Clear all caches."""
        if hasattr(self, 'redis_client'):
            # Clear Redis cache
            self.redis_client.flushdb()
        
        # Clear LRU cache
        self._lru_cache.cache_clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "lru_cache_size": len(self._lru_cache.cache_info()),
            "lru_cache_hits": self._lru_cache.cache_info().hits,
            "lru_cache_misses": self._lru_cache.cache_info().misses
        }
        
        if hasattr(self, 'redis_client'):
            # Get Redis stats
            info = self.redis_client.info()
            stats.update({
                "redis_used_memory": info["used_memory"],
                "redis_connected_clients": info["connected_clients"],
                "redis_commands_processed": info["total_commands_processed"]
            })
        
        return stats 