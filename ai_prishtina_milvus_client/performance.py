"""
Performance optimization features for Milvus operations including caching, batching, and parallel processing with async support.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable, TypeVar, Awaitable
import numpy as np
from pydantic import BaseModel, Field
import logging
from datetime import datetime, timedelta
import asyncio
import time
from functools import lru_cache
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

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
    use_asyncio: bool = Field(True, description="Whether to use asyncio")
    use_threading: bool = Field(False, description="Whether to use threading")
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
        self.cache_lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the performance optimizer."""
        if self.config.cache_config.enabled:
            self._cleanup_task = asyncio.create_task(self._cleanup_cache())
            
    async def stop(self):
        """Stop the performance optimizer."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    def cached(self, func: Callable[..., Union[T, Awaitable[T]]]) -> Callable[..., Awaitable[T]]:
        """
        Decorator to cache function results.
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        if not self.config.cache_config.enabled:
            return func
            
        async def wrapper(*args, **kwargs) -> T:
            # Generate cache key
            key = self._generate_cache_key(func, args, kwargs)
            
            # Check cache
            async with self.cache_lock:
                if key in self.cache:
                    entry = self.cache[key]
                    if not entry.is_expired():
                        return entry.value
                    del self.cache[key]
                    
            # Call function
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            
            # Cache result
            async with self.cache_lock:
                if len(self.cache) >= self.config.cache_config.max_size:
                    await self._evict_cache()
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
        
    async def _evict_cache(self) -> None:
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
                
    async def _cleanup_cache(self) -> None:
        """Periodically clean up expired cache entries."""
        while True:
            await asyncio.sleep(self.config.cache_config.cleanup_interval)
            async with self.cache_lock:
                await self._evict_cache()
                
    async def batch_process(
        self,
        items: List[Any],
        func: Callable[[Any], Union[T, Awaitable[T]]],
        chunk_size: Optional[int] = None
    ) -> List[T]:
        """
        Process items in batches asynchronously.
        
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
            chunk_results = await asyncio.gather(
                *(self._process_item(func, item) for item in chunk)
            )
            results.extend(chunk_results)
            
        return results
        
    async def _process_item(
        self,
        func: Callable[[Any], Union[T, Awaitable[T]]],
        item: Any
    ) -> T:
        """Process a single item."""
        result = func(item)
        if asyncio.iscoroutine(result):
            result = await result
        return result
        
    async def parallel_map(
        self,
        items: List[Any],
        func: Callable[[Any], Union[T, Awaitable[T]]],
        chunk_size: Optional[int] = None
    ) -> List[T]:
        """
        Map function over items in parallel asynchronously.
        
        Args:
            items: List of items
            func: Function to apply
            chunk_size: Optional chunk size
            
        Returns:
            List of results
        """
        if not items:
            return []
            
        if not self.config.use_asyncio:
            return await asyncio.gather(
                *(self._process_item(func, item) for item in items)
            )
            
        chunk_size = chunk_size or self.config.batch_config.batch_size
        results = []
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            chunk_results = await asyncio.gather(
                *(self._process_item(func, item) for item in chunk)
            )
            results.extend(chunk_results)
            
        return results
        
    async def optimize_vector_operations(
        self,
        vectors: Union[np.ndarray, List[List[float]]],
        operation: str,
        other_vectors: Optional[Union[np.ndarray, List[List[float]]]] = None
    ) -> Union[np.ndarray, float]:
        """
        Optimize vector operations asynchronously.
        
        Args:
            vectors: Input vectors
            operation: Operation to perform
            other_vectors: Optional other vectors for binary operations
            
        Returns:
            Operation result
        """
        # Convert to numpy array if needed
        if not isinstance(vectors, np.ndarray):
            vectors = np.array(vectors)
        if other_vectors is not None and not isinstance(other_vectors, np.ndarray):
            other_vectors = np.array(other_vectors)
            
        # Perform operation
        if operation == "normalize":
            return await asyncio.to_thread(
                lambda: vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            )
        elif operation == "dot":
            if other_vectors is None:
                raise ValueError("Other vectors required for dot product")
            return await asyncio.to_thread(
                lambda: np.dot(vectors, other_vectors.T)
            )
        elif operation == "cosine":
            if other_vectors is None:
                raise ValueError("Other vectors required for cosine similarity")
            return await asyncio.to_thread(
                lambda: np.dot(vectors, other_vectors.T) / (
                    np.linalg.norm(vectors, axis=1, keepdims=True) *
                    np.linalg.norm(other_vectors, axis=1, keepdims=True).T
                )
            )
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
    async def profile_operation(
        self,
        func: Callable[..., Union[T, Awaitable[T]]],
        *args,
        **kwargs
    ) -> Tuple[T, Dict[str, Any]]:
        """
        Profile operation execution asynchronously.
        
        Args:
            func: Function to profile
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Tuple of (result, metrics)
        """
        start_time = time.time()
        start_memory = await self._get_memory_usage()
        
        # Execute function
        result = func(*args, **kwargs)
        if asyncio.iscoroutine(result):
            result = await result
            
        end_time = time.time()
        end_memory = await self._get_memory_usage()
        
        # Calculate metrics
        metrics = {
            "execution_time": end_time - start_time,
            "memory_usage": end_memory - start_memory,
            "timestamp": datetime.now().isoformat()
        }
        
        return result, metrics
        
    async def _get_memory_usage(self) -> int:
        """
        Get current memory usage in bytes.
        
        Returns:
            Memory usage in bytes
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss 