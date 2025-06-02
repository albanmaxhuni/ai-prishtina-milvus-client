"""
Unit tests for batch operations module.
"""

import pytest
from unittest.mock import Mock, patch
from ai_prishtina_milvus_client import BatchProcessor, BatchConfig, BatchMetrics

@pytest.fixture
def mock_client():
    """Create mock Milvus client."""
    client = Mock()
    client.insert = Mock(return_value=None)
    client.delete = Mock(return_value=None)
    client.search = Mock(return_value=[{"id": i, "score": 0.9} for i in range(10)])
    return client

def test_batch_insert(mock_client, sample_vectors, sample_metadata, batch_config):
    """Test batch insert operation."""
    processor = BatchProcessor(mock_client, batch_config)
    
    # Test successful insert
    metrics = processor.batch_insert(sample_vectors, sample_metadata)
    assert isinstance(metrics, BatchMetrics)
    assert metrics.total_items == len(sample_vectors)
    assert metrics.successful_items == len(sample_vectors)
    assert metrics.failed_items == 0
    assert metrics.total_time > 0
    assert metrics.average_time_per_item > 0
    
    # Test insert with error
    mock_client.insert.side_effect = Exception("Insert failed")
    metrics = processor.batch_insert(sample_vectors, sample_metadata)
    assert metrics.failed_items == len(sample_vectors)
    assert len(metrics.errors) > 0
    
def test_batch_delete(mock_client, batch_config):
    """Test batch delete operation."""
    processor = BatchProcessor(mock_client, batch_config)
    
    # Test successful delete
    expr = "id > 0"
    metrics = processor.batch_delete(expr)
    assert isinstance(metrics, BatchMetrics)
    assert metrics.total_items > 0
    assert metrics.successful_items > 0
    assert metrics.failed_items == 0
    
    # Test delete with error
    mock_client.delete.side_effect = Exception("Delete failed")
    metrics = processor.batch_delete(expr)
    assert metrics.failed_items > 0
    assert len(metrics.errors) > 0
    
def test_batch_search(mock_client, sample_vectors, batch_config):
    """Test batch search operation."""
    processor = BatchProcessor(mock_client, batch_config)
    
    # Test successful search
    results, metrics = processor.batch_search(sample_vectors)
    assert isinstance(metrics, BatchMetrics)
    assert len(results) == len(sample_vectors)
    assert metrics.total_items == len(sample_vectors)
    assert metrics.successful_items == len(sample_vectors)
    assert metrics.failed_items == 0
    
    # Test search with error
    mock_client.search.side_effect = Exception("Search failed")
    results, metrics = processor.batch_search(sample_vectors)
    assert metrics.failed_items == len(sample_vectors)
    assert len(metrics.errors) > 0
    
def test_process_batch(mock_client, sample_vectors, sample_metadata, batch_config):
    """Test single batch processing."""
    processor = BatchProcessor(mock_client, batch_config)
    
    # Test successful batch
    result = processor._process_batch(sample_vectors, sample_metadata)
    assert result["successful"] == len(sample_vectors)
    assert result["failed"] == 0
    assert len(result["errors"]) == 0
    
    # Test batch with error
    mock_client.insert.side_effect = Exception("Insert failed")
    result = processor._process_batch(sample_vectors, sample_metadata)
    assert result["successful"] == 0
    assert result["failed"] == len(sample_vectors)
    assert len(result["errors"]) > 0
    
def test_batch_metrics():
    """Test batch metrics."""
    metrics = BatchMetrics(
        total_items=100,
        successful_items=90,
        failed_items=10,
        total_time=1.0,
        average_time_per_item=0.01,
        errors=[{"error": "Test error"}]
    )
    
    assert metrics.total_items == 100
    assert metrics.successful_items == 90
    assert metrics.failed_items == 10
    assert metrics.total_time == 1.0
    assert metrics.average_time_per_item == 0.01
    assert len(metrics.errors) == 1 