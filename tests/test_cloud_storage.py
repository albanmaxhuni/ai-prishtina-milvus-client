"""Unit tests for cloud storage integration."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import boto3
import pytest
from google.cloud import storage
from azure.storage.blob import BlobServiceClient

from ai_prishtina_milvus_client.cloud_storage import (
    CloudStorageConfig,
    CloudStorageFactory,
    load_cloud_storage,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "vectors": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "metadata": [
            {"category": "A", "score": 0.8},
            {"category": "B", "score": 0.9},
        ],
    }


@pytest.fixture
def aws_config():
    """Create AWS S3 configuration."""
    return CloudStorageConfig(
        service="s3",
        bucket_name="test-bucket",
        region_name="us-east-1",
        aws_access_key_id="test-key",
        aws_secret_access_key="test-secret",
    )


@pytest.fixture
def gcp_config():
    """Create GCP configuration."""
    return CloudStorageConfig(
        service="gcp",
        bucket_name="test-bucket",
        project_id="test-project",
        credentials_path="test-credentials.json",
    )


@pytest.fixture
def azure_config():
    """Create Azure configuration."""
    return CloudStorageConfig(
        service="azure",
        container_name="test-container",
        account_name="test-account",
        account_key="test-key",
    )


@patch("boto3.client")
def test_aws_s3_download(mock_boto3, aws_config, sample_data):
    """Test downloading from AWS S3."""
    # Mock S3 client
    mock_s3 = MagicMock()
    mock_boto3.return_value = mock_s3
    
    # Create temporary file with sample data
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(str(sample_data).encode())
        temp_path = temp_file.name
    
    # Mock S3 download
    mock_s3.download_file.return_value = None
    
    # Create AWS S3 client
    client = CloudStorageFactory.create(aws_config)
    
    # Test download
    local_path = client.download("test-file.json")
    assert os.path.exists(local_path)
    
    # Clean up
    os.unlink(temp_path)
    os.unlink(local_path)


@patch("google.cloud.storage.Client")
def test_gcp_download(mock_gcp, gcp_config, sample_data):
    """Test downloading from GCP."""
    # Mock GCP client
    mock_client = MagicMock()
    mock_gcp.return_value = mock_client
    
    # Mock bucket and blob
    mock_bucket = MagicMock()
    mock_client.bucket.return_value = mock_bucket
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    
    # Create temporary file with sample data
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(str(sample_data).encode())
        temp_path = temp_file.name
    
    # Mock GCP download
    mock_blob.download_to_filename.return_value = None
    
    # Create GCP client
    client = CloudStorageFactory.create(gcp_config)
    
    # Test download
    local_path = client.download("test-file.json")
    assert os.path.exists(local_path)
    
    # Clean up
    os.unlink(temp_path)
    os.unlink(local_path)


@patch("azure.storage.blob.BlobServiceClient")
def test_azure_download(mock_azure, azure_config, sample_data):
    """Test downloading from Azure."""
    # Mock Azure client
    mock_client = MagicMock()
    mock_azure.return_value = mock_client
    
    # Mock container and blob
    mock_container = MagicMock()
    mock_client.get_container_client.return_value = mock_container
    mock_blob = MagicMock()
    mock_container.get_blob_client.return_value = mock_blob
    
    # Create temporary file with sample data
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file.write(str(sample_data).encode())
        temp_path = temp_file.name
    
    # Mock Azure download
    mock_blob.download_blob.return_value.readall.return_value = str(sample_data).encode()
    
    # Create Azure client
    client = CloudStorageFactory.create(azure_config)
    
    # Test download
    local_path = client.download("test-file.json")
    assert os.path.exists(local_path)
    
    # Clean up
    os.unlink(temp_path)
    os.unlink(local_path)


def test_invalid_cloud_storage_type():
    """Test invalid cloud storage type."""
    config = CloudStorageConfig(
        service="invalid",
        bucket_name="test-bucket",
    )
    
    with pytest.raises(ValueError):
        CloudStorageFactory.create(config)


def test_missing_credentials():
    """Test missing credentials."""
    # AWS without credentials
    config = CloudStorageConfig(
        service="s3",
        bucket_name="test-bucket",
    )
    
    with pytest.raises(ValueError):
        CloudStorageFactory.create(config)
    
    # GCP without credentials
    config = CloudStorageConfig(
        service="gcp",
        bucket_name="test-bucket",
    )
    
    with pytest.raises(ValueError):
        CloudStorageFactory.create(config)
    
    # Azure without credentials
    config = CloudStorageConfig(
        service="azure",
        container_name="test-container",
    )
    
    with pytest.raises(ValueError):
        CloudStorageFactory.create(config)


def test_load_cloud_storage():
    """Test loading cloud storage configuration."""
    # Create temporary config file
    config_data = {
        "service": "s3",
        "bucket_name": "test-bucket",
        "region_name": "us-east-1",
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
    }
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_file.write(str(config_data).encode())
        config_path = temp_file.name
    
    # Test loading
    with patch("boto3.client") as mock_boto3:
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        
        client = load_cloud_storage(config_path)
        assert client is not None
    
    # Clean up
    os.unlink(config_path) 