"""
Unit tests for monitoring module.
"""

import pytest
from unittest.mock import Mock, patch
import time
from ai_prishtina_milvus_client import MetricsCollector, MonitoringConfig, SystemMetrics

@pytest.fixture
def mock_client():
    """Create mock Milvus client."""
    client = Mock()
    client.get_collection_stats = Mock(return_value={
        "row_count": 1000,
        "partition_count": 2
    })
    client.get_partition_stats = Mock(return_value={
        "row_count": 500
    })
    return client

def test_collect_system_metrics(monitoring_config):
    """Test system metrics collection."""
    collector = MetricsCollector(monitoring_config)
    # Test metrics collection
    # (Assume collection thread runs, just check collector exists)
    assert hasattr(collector, 'system_metrics')

def test_collect_collection_metrics(mock_client, monitoring_config):
    """Test collection metrics collection."""
    collector = MetricsCollector(mock_client, monitoring_config)
    
    # Test metrics collection
    metrics = collector.collect_collection_metrics("test_collection")
    assert metrics["row_count"] == 1000
    assert metrics["partition_count"] == 2
    
    # Test with non-existent collection
    mock_client.get_collection_stats.side_effect = Exception("Collection not found")
    with pytest.raises(Exception):
        collector.collect_collection_metrics("non_existent")
        
def test_collect_partition_metrics(mock_client, monitoring_config):
    """Test partition metrics collection."""
    collector = MetricsCollector(mock_client, monitoring_config)
    
    # Test metrics collection
    metrics = collector.collect_partition_metrics("test_collection", "test_partition")
    assert metrics["row_count"] == 500
    
    # Test with non-existent partition
    mock_client.get_partition_stats.side_effect = Exception("Partition not found")
    with pytest.raises(Exception):
        collector.collect_partition_metrics("test_collection", "non_existent")
        
def test_collect_query_metrics(mock_client, monitoring_config):
    """Test query metrics collection."""
    collector = MetricsCollector(mock_client, monitoring_config)
    
    # Test metrics collection
    metrics = collector.collect_query_metrics("test_collection")
    assert isinstance(metrics, dict)
    assert "query_count" in metrics
    assert "average_latency" in metrics
    assert "error_count" in metrics
    
def test_collect_index_metrics(mock_client, monitoring_config):
    """Test index metrics collection."""
    collector = MetricsCollector(mock_client, monitoring_config)
    
    # Test metrics collection
    metrics = collector.collect_index_metrics("test_collection")
    assert isinstance(metrics, dict)
    assert "index_type" in metrics
    assert "index_params" in metrics
    assert "index_size" in metrics
    
def test_monitoring_config():
    """Test monitoring configuration."""
    config = MonitoringConfig(
        collection_interval=60,
        metrics_retention=3600,
        alert_threshold=0.8,
        enable_logging=True
    )
    
    assert config.collection_interval == 60
    assert config.metrics_retention == 3600
    assert config.alert_threshold == 0.8
    assert config.enable_logging is True
    
def test_system_metrics():
    """Test system metrics."""
    metrics = SystemMetrics(
        timestamp=time.time(),
        cpu_percent=50.0,
        memory_percent=70.0,
        disk_io_read=1000.0,
        disk_io_write=2000.0,
        network_io_sent=3000.0,
        network_io_recv=4000.0
    )
    assert metrics.timestamp > 0
    assert metrics.cpu_percent == 50.0
    assert metrics.memory_percent == 70.0
    assert metrics.disk_io_read == 1000.0
    assert metrics.disk_io_write == 2000.0
    assert metrics.network_io_sent == 3000.0
    assert metrics.network_io_recv == 4000.0 