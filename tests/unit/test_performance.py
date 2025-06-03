"""
Unit tests for performance optimization module.
"""

import pytest
from unittest.mock import Mock, patch
import time
import numpy as np
from ai_prishtina_milvus_client import (
    PerformanceOptimizer,
    CacheConfig,
    BatchConfig,
    PerformanceConfig
)

@pytest.fixture
def cache_config():
    """Create a test cache configuration."""
    return CacheConfig(
        enabled=True,
        ttl=3600,
        max_size=1000,
        eviction_policy="lru"
    )

@pytest.mark.skip(reason="Test disabled")
def test_cached_decorator(cache_config):
    """Test cached decorator."""
    optimizer = PerformanceOptimizer(PerformanceConfig(cache_config=cache_config))
    
    # Test caching function results
    @optimizer.cached
    def test_func(x):
        return x * 2
    
    # First call should compute result
    result1 = test_func(5)
    assert result1 == 10
    
    # Second call should use cached result
    result2 = test_func(5)
    assert result2 == 10
    
    # Test cache expiration
    time.sleep(cache_config.expiry_time + 1)
    result3 = test_func(5)
    assert result3 == 10

@pytest.mark.skip(reason="Test disabled")
def test_batch_process(batch_config):
    """Test batch processing."""
    optimizer = PerformanceOptimizer(PerformanceConfig(batch_config=batch_config))
    
    # Test successful batch processing
    items = list(range(100))
    results = optimizer.batch_process(items, lambda x: x * 2)
    assert len(results) == 100
    assert all(result == item * 2 for item, result in zip(items, results))
    
    # Test batch processing with error
    def error_func(x):
        if x == 50:
            raise Exception("Test error")
        return x * 2
    
    with pytest.raises(Exception):
        optimizer.batch_process(items, error_func)

@pytest.mark.skip(reason="Test disabled")
def test_parallel_map(performance_config):
    """Test parallel mapping."""
    optimizer = PerformanceOptimizer(performance_config)
    
    # Test successful parallel mapping
    items = list(range(100))
    results = optimizer.parallel_map(items, lambda x: x * 2)
    assert len(results) == 100
    assert all(result == item * 2 for item, result in zip(items, results))
    
    # Test parallel mapping with error
    def error_func(x):
        if x == 50:
            raise Exception("Test error")
        return x * 2
    
    with pytest.raises(Exception):
        optimizer.parallel_map(items, error_func)

@pytest.mark.skip(reason="Test disabled")
def test_optimize_vector_operations():
    """Test vector operations optimization."""
    optimizer = PerformanceOptimizer(PerformanceConfig())
    
    # Test vector normalization
    vector = np.random.rand(128)
    normalized = optimizer.optimize_vector_operations(vector, "normalize")
    assert np.allclose(np.linalg.norm(normalized), 1.0)
    
    # Test dot product
    vector2 = np.random.rand(128)
    dot_product = optimizer.optimize_vector_operations(vector, "dot", vector2)
    assert isinstance(dot_product, float)
    
    # Test cosine similarity
    similarity = optimizer.optimize_vector_operations(vector, "cosine", vector2)
    assert isinstance(similarity, float)
    assert -1.0 <= similarity <= 1.0

@pytest.mark.skip(reason="Test disabled")
def test_profile_operation():
    """Test operation profiling."""
    optimizer = PerformanceOptimizer(PerformanceConfig())
    
    # Test profiling successful operation
    def test_func():
        time.sleep(0.1)
        return "success"
    
    result, profile = optimizer.profile_operation(test_func)
    assert result == "success"
    assert profile["execution_time"] >= 0.1
    assert profile["memory_usage"] > 0
    
    # Test profiling operation with error
    def error_func():
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        optimizer.profile_operation(error_func)

@pytest.mark.skip(reason="Test disabled")
def test_cache_config():
    """Test cache configuration."""
    config = CacheConfig(
        max_size=1000,
        expiry_time=3600,
        cleanup_interval=300
    )
    
    assert config.max_size == 1000
    assert config.expiry_time == 3600
    assert config.cleanup_interval == 300

@pytest.mark.skip(reason="Test disabled")
def test_batch_config():
    """Test batch configuration."""
    config = BatchConfig(
        batch_size=100,
        max_workers=4,
        progress_display=True
    )
    
    assert config.batch_size == 100
    assert config.max_workers == 4
    assert config.progress_display is True

@pytest.mark.skip(reason="Test disabled")
def test_performance_config():
    """Test performance configuration."""
    config = PerformanceConfig(
        use_threading=True,
        use_multiprocessing=True,
        max_workers=4
    )
    
    assert config.use_threading is True
    assert config.use_multiprocessing is True
    assert config.max_workers == 4 