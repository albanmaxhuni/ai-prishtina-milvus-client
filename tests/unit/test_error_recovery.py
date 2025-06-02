"""
Unit tests for error recovery module.
"""

import pytest
from unittest.mock import Mock, patch
import time
from ai_prishtina_milvus_client import ErrorRecovery, RetryConfig, BackupConfig

@pytest.fixture
def mock_client():
    """Create mock Milvus client."""
    client = Mock()
    client.create_collection = Mock(return_value=None)
    client.drop_collection = Mock(return_value=None)
    client.insert = Mock(return_value=None)
    client.search = Mock(return_value=[{"id": i, "score": 0.9} for i in range(10)])
    return client

def test_retry_operation(mock_client, retry_config):
    """Test retry operation."""
    recovery = ErrorRecovery(mock_client, retry_config)
    
    # Test successful operation
    result = recovery.retry_operation(lambda: "success")
    assert result == "success"
    
    # Test operation with retries
    mock_func = Mock(side_effect=[Exception("Error"), Exception("Error"), "success"])
    result = recovery.retry_operation(mock_func)
    assert result == "success"
    assert mock_func.call_count == 3
    
    # Test operation with max retries exceeded
    mock_func = Mock(side_effect=Exception("Error"))
    with pytest.raises(Exception):
        recovery.retry_operation(mock_func)
    assert mock_func.call_count == retry_config.max_retries + 1

def test_backup_collection(mock_client, backup_config):
    """Test collection backup."""
    recovery = ErrorRecovery(mock_client, backup_config)
    
    # Test successful backup
    backup_path = recovery.backup_collection("test_collection")
    assert backup_path is not None
    assert backup_path.startswith(backup_config.backup_dir)
    
    # Test backup with error
    mock_client.get_collection_stats.side_effect = Exception("Backup failed")
    with pytest.raises(Exception):
        recovery.backup_collection("test_collection")

def test_restore_collection(mock_client, backup_config):
    """Test collection restore."""
    recovery = ErrorRecovery(mock_client, backup_config)
    
    # Test successful restore
    recovery.restore_collection("test_collection", "backup_path")
    mock_client.create_collection.assert_called_once()
    mock_client.insert.assert_called()
    
    # Test restore with error
    mock_client.create_collection.side_effect = Exception("Restore failed")
    with pytest.raises(Exception):
        recovery.restore_collection("test_collection", "backup_path")

def test_handle_connection_error(mock_client, retry_config):
    """Test connection error handling."""
    recovery = ErrorRecovery(mock_client, retry_config)
    
    # Test successful reconnection
    mock_client.connect = Mock(return_value=True)
    result = recovery.handle_connection_error()
    assert result is True
    mock_client.connect.assert_called_once()
    
    # Test reconnection with error
    mock_client.connect.side_effect = Exception("Connection failed")
    with pytest.raises(Exception):
        recovery.handle_connection_error()

def test_retry_config():
    """Test retry configuration."""
    config = RetryConfig(
        max_retries=3,
        retry_delay=1.0,
        exponential_backoff=True,
        max_delay=10.0
    )
    
    assert config.max_retries == 3
    assert config.retry_delay == 1.0
    assert config.exponential_backoff is True
    assert config.max_delay == 10.0

def test_backup_config():
    """Test backup configuration."""
    config = BackupConfig(
        backup_dir="/tmp/backups",
        backup_interval=3600,
        max_backups=5,
        compression=True
    )
    
    assert config.backup_dir == "/tmp/backups"
    assert config.backup_interval == 3600
    assert config.max_backups == 5
    assert config.compression is True 