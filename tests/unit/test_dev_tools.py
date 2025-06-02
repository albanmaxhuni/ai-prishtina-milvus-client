"""
Unit tests for development tools module.
"""

import pytest
from unittest.mock import Mock, patch
import time
import logging
from ai_prishtina_milvus_client import (
    DevTools,
    LoggingConfig,
    DebugConfig,
    TestConfig
)

def test_debug_decorator(debug_config):
    """Test debug decorator."""
    tools = DevTools(debug_config)
    
    # Test successful function execution
    @tools.debug
    def test_func(x):
        return x * 2
    
    result = test_func(5)
    assert result == 10
    
    # Test function with error
    @tools.debug
    def error_func():
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        error_func()

def test_profile_decorator():
    """Test profile decorator."""
    tools = DevTools()
    
    # Test successful function execution
    @tools.profile
    def test_func():
        time.sleep(0.1)
        return "success"
    
    result, profile = test_func()
    assert result == "success"
    assert profile["execution_time"] >= 0.1
    assert profile["memory_usage"] > 0
    
    # Test function with error
    @tools.profile
    def error_func():
        raise Exception("Test error")
    
    with pytest.raises(Exception):
        error_func()

def test_run_tests(test_config):
    """Test test runner."""
    tools = DevTools(test_config)
    
    # Test running tests
    results = tools.run_tests()
    assert isinstance(results, dict)
    assert "passed" in results
    assert "failed" in results
    assert "skipped" in results
    
    # Test running specific test file
    results = tools.run_tests(test_file="test_example.py")
    assert isinstance(results, dict)

def test_generate_test_data():
    """Test test data generation."""
    tools = DevTools()
    
    # Test generating test data
    data = tools.generate_test_data(
        num_vectors=10,
        vector_dim=128,
        metadata_fields=["id", "text", "category"]
    )
    assert len(data) == 10
    assert all("vector" in item for item in data)
    assert all("metadata" in item for item in data)
    assert all(len(item["vector"]) == 128 for item in data)
    assert all(all(field in item["metadata"] for field in ["id", "text", "category"]) for item in data)

def test_validate_test_results():
    """Test test result validation."""
    tools = DevTools()
    
    # Test successful validation
    expected = [{"id": i, "score": 0.9} for i in range(5)]
    actual = [{"id": i, "score": 0.9} for i in range(5)]
    assert tools.validate_test_results(actual, expected)
    
    # Test validation with tolerance
    expected = [{"id": i, "score": 0.9} for i in range(5)]
    actual = [{"id": i, "score": 0.9001} for i in range(5)]
    assert tools.validate_test_results(actual, expected, tolerance=0.001)
    
    # Test validation with different results
    expected = [{"id": i, "score": 0.9} for i in range(5)]
    actual = [{"id": i, "score": 0.8} for i in range(5)]
    assert not tools.validate_test_results(actual, expected)

def test_create_test_collection(mock_client):
    """Test test collection creation."""
    tools = DevTools()
    
    # Test successful collection creation
    collection = tools.create_test_collection(
        mock_client,
        "test_collection",
        vector_dim=128,
        index_type="IVF_FLAT"
    )
    assert collection is not None
    mock_client.create_collection.assert_called_once()
    
    # Test collection creation with error
    mock_client.create_collection.side_effect = Exception("Creation failed")
    with pytest.raises(Exception):
        tools.create_test_collection(mock_client, "test_collection")

def test_cleanup_test_collection(mock_client):
    """Test test collection cleanup."""
    tools = DevTools()
    
    # Test successful cleanup
    tools.cleanup_test_collection(mock_client, "test_collection")
    mock_client.drop_collection.assert_called_once_with("test_collection")
    
    # Test cleanup with error
    mock_client.drop_collection.side_effect = Exception("Cleanup failed")
    with pytest.raises(Exception):
        tools.cleanup_test_collection(mock_client, "test_collection")

def test_get_function_info():
    """Test function information retrieval."""
    tools = DevTools()
    
    def test_func(x: int, y: str = "test") -> bool:
        """Test function docstring."""
        return True
    
    info = tools.get_function_info(test_func)
    assert info["name"] == "test_func"
    assert info["module"] == __name__
    assert info["docstring"] == "Test function docstring."
    assert "x" in info["signature"]
    assert "y" in info["signature"]
    assert info["is_async"] is False
    assert info["is_generator"] is False

def test_logging_config():
    """Test logging configuration."""
    config = LoggingConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        file_path="test.log",
        max_size=1024,
        backup_count=5
    )
    
    assert config.level == logging.INFO
    assert config.format is not None
    assert config.file_path == "test.log"
    assert config.max_size == 1024
    assert config.backup_count == 5

def test_debug_config():
    """Test debug configuration."""
    config = DebugConfig(
        enabled=True,
        break_on_error=True,
        log_level=logging.DEBUG,
        trace_calls=True
    )
    
    assert config.enabled is True
    assert config.break_on_error is True
    assert config.log_level == logging.DEBUG
    assert config.trace_calls is True

def test_test_config():
    """Test test configuration."""
    config = TestConfig(
        test_dir="tests",
        file_pattern="test_*.py",
        collect_coverage=True,
        parallel_execution=True
    )
    
    assert config.test_dir == "tests"
    assert config.file_pattern == "test_*.py"
    assert config.collect_coverage is True
    assert config.parallel_execution is True 