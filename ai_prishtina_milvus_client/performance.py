"""
Performance optimization features for Milvus operations including caching, batching, and parallel processing.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable, TypeVar
import numpy as np
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import threading
import queue
import time
from functools import lru_cache
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

T = TypeVar('T')

class CacheConfig(BaseModel):
    """Configuration for caching."""
    max_size: int = Field(1000, description="Maximum cache size")
    ttl: int = Field(3600, description="Time to live in seconds")
    expiry_time: int = Field(3600, description="Expiry time (alias for ttl)")
    cleanup_interval: int = Field(300, description="Cache cleanup interval in seconds")
    enabled: bool = Field(True, description="Whether caching is enabled")

class BatchConfig(BaseModel):
    """Configuration for batching."""
    batch_size: int = Field(1000, description="Batch size")
    max_workers: int = Field(4, description="Maximum number of workers")
    progress_display: bool = Field(True, description="Whether to show progress bar")
    timeout: int = Field(30, description="Batch operation timeout in seconds")
    validation_config: Optional[Any] = Field(None, description="Vector validation config")

class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    use_threading: bool = Field(True, description="Whether to use threading")
    use_multiprocessing: bool = Field(False, description="Whether to use multiprocessing")
    max_workers: int = Field(4, description="Maximum number of workers")
    cache_config: CacheConfig = Field(default_factory=CacheConfig)
    batch_config: BatchConfig = Field(default_factory=BatchConfig)

class CacheEntry:
    """Cache entry with timestamp."""
    
    def __init__(self, value: Any, ttl: int):
        """
        Initialize cache entry.
        
        Args:
            value: Cached value
            ttl: Time to live in seconds
        """
        self.value = value
        self.timestamp = datetime.now()
        self.ttl = ttl
        
    def is_expired(self) -> bool:
        """
        Check if entry is expired.
        
        Returns:
            True if expired, False otherwise
        """
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl

class PerformanceOptimizer:
    """Optimizer for performance features."""
    
    def __init__(self, config: PerformanceConfig):
        """
        Initialize performance optimizer.
        
        Args:
            config: Performance configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.Lock()
        
    def cached(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        if not self.config.cache_config.enabled:
            return func
            
        def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = self._generate_cache_key(func, args, kwargs)
            
            # Check cache
            with self.cache_lock:
                if key in self.cache:
                    entry = self.cache[key]
                    if not entry.is_expired():
                        return entry.value
                    del self.cache[key]
                    
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            with self.cache_lock:
                if len(self.cache) >= self.config.cache_config.max_size:
                    self._evict_cache()
                self.cache[key] = CacheEntry(
                    result,
                    self.config.cache_config.ttl
                )
                
            return result
            
        return wrapper
        
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """
        Generate cache key for function call.
        
        Args:
            func: Function
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        # Convert arguments to string
        args_str = json.dumps(args, sort_keys=True)
        kwargs_str = json.dumps(kwargs, sort_keys=True)
        
        # Generate hash
        key = f"{func.__name__}:{args_str}:{kwargs_str}"
        return hashlib.md5(key.encode()).hexdigest()
        
    def _evict_cache(self) -> None:
        """Evict expired entries from cache."""
        now = datetime.now()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
            
        # If still full, remove oldest entries
        if len(self.cache) >= self.config.cache_config.max_size:
            sorted_entries = sorted(
                self.cache.items(),
                key=lambda x: x[1].timestamp
            )
            for key, _ in sorted_entries[:len(self.cache) // 2]:
                del self.cache[key]
                
    def batch_process(
        self,
        items: List[Any],
        func: Callable[[Any], T],
        chunk_size: Optional[int] = None
    ) -> List[T]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            func: Function to apply to each item
            chunk_size: Optional chunk size
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        chunk_size = chunk_size or self.config.batch_config.batch_size
        results = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_results = [func(item) for item in chunk]
            results.extend(chunk_results)
            
        return results
        
    def _process_batch_threaded(
        self,
        batch: List[Any],
        process_func: Callable[[Any], T]
    ) -> List[T]:
        """
        Process batch using threads.
        
        Args:
            batch: List of items to process
            process_func: Function to process each item
            
        Returns:
            List of processed results
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.config.batch_config.max_workers) as executor:
            future_to_item = {
                executor.submit(process_func, item): item
                for item in batch
            }
            
            for future in as_completed(future_to_item):
                try:
                    result = future.result(timeout=self.config.batch_config.timeout)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch processing failed: {str(e)}")
                    raise
                    
        return results
        
    def parallel_map(
        self,
        items: List[Any],
        func: Callable[[Any], T],
        chunk_size: Optional[int] = None
    ) -> List[T]:
        """
        Map function over items in parallel.
        
        Args:
            items: List of items
            func: Function to apply
            chunk_size: Optional chunk size
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        if not self.config.use_threading:
            return [func(item) for item in items]
            
        chunk_size = chunk_size or self.config.batch_config.batch_size
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.batch_config.max_workers) as executor:
            futures = []
            for item in items:
                futures.append(executor.submit(func, item))
                
            for future in futures:
                results.append(future.result())
                
        return results
        
    def optimize_vector_operations(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        operation: str,
        other_vectors: Optional[Union[np.ndarray, List[List[float]]]] = None
    ) -> Union[np.ndarray, float]:
        """
        Optimize vector operations.
        
        Args:
            vectors: Input vectors
            operation: Operation to perform
            other_vectors: Optional other vectors for binary operations
            
        Returns:
            Operation result
        """
        if isinstance(vectors, list):
            vectors = np.array(vectors)
            
        if other_vectors is not None and isinstance(other_vectors, list):
            other_vectors = np.array(other_vectors)
            
        if operation == "normalize":
            # Handle single vector
            if vectors.ndim == 1:
                norm = np.linalg.norm(vectors)
                if norm > 0:
                    return vectors / norm
                return vectors
                
            # Handle multiple vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return np.divide(vectors, norms, where=norms > 0)
            
        elif operation == "dot":
            if other_vectors is None:
                raise ValueError("Other vectors required for dot product")
            return np.dot(vectors, other_vectors)
            
        elif operation == "cosine":
            if other_vectors is None:
                raise ValueError("Other vectors required for cosine similarity")
            # Normalize vectors
            v1 = self.optimize_vector_operations(vectors, "normalize")
            v2 = self.optimize_vector_operations(other_vectors, "normalize")
            return np.dot(v1, v2)
            
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    def profile_operation(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Profile an operation's execution time and memory usage."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile = {
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "success": True
            }
            return result, profile
            
        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            profile = {
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "success": False,
                "error": str(e)
            }
            raise
            
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            return 0 