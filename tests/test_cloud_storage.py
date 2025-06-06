"""Tests for cloud storage operations."""

import pytest
import asyncio
from typing import List, Dict, Any
import json
from unittest.mock import AsyncMock, patch

from ai_prishtina_milvus_client.cloud_storage import CloudStorage, CloudStorageConfig
from ai_prishtina_milvus_client.config import MilvusConfig
from ai_prishtina_milvus_client.exceptions import CloudStorageError


@pytest.fixture
def storage_config() -> CloudStorageConfig:
    """Create cloud storage configuration."""
    return CloudStorageConfig(
        provider="aws",
        bucket_name="test-bucket",
        region="us-west-2",
        credentials_path="test-credentials.json",
        prefix="test/",
        max_retries=3,
        retry_delay=1.0,
        timeout=30.0
    )


@pytest.fixture
async def cloud_storage(milvus_config: MilvusConfig, storage_config: CloudStorageConfig):
    """Create cloud storage instance."""
    storage = CloudStorage(milvus_config, storage_config)
    yield storage
    await storage.cleanup()


@pytest.fixture
def test_collection_data() -> Dict[str, Any]:
    """Generate test collection data."""
    return {
        "vectors": [[0.1, 0.2, 0.3] for _ in range(10)],
        "metadata": [{"id": i} for i in range(10)]
    }


@pytest.mark.asyncio
async def test_upload_collection(
    cloud_storage: CloudStorage,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test upload collection."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.upload_file.return_value = None
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        # Upload collection
        result = await cloud_storage.upload_collection(
            collection_name=test_collection,
            data=test_collection_data
        )
        
        # Verify upload
        mock_s3.upload_file.assert_called_once()
        assert result["status"] == "success"
        assert result["uploaded_files"] == 2  # vectors and metadata


@pytest.mark.asyncio
async def test_download_collection(
    cloud_storage: CloudStorage,
    test_collection: str
):
    """Test download collection."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.download_file.return_value = None
    
    # Mock file content
    mock_content = {
        "vectors": [[0.1, 0.2, 0.3] for _ in range(10)],
        "metadata": [{"id": i} for i in range(10)]
    }
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        with patch("aiofiles.open", AsyncMock()) as mock_file:
            mock_file.return_value.__aenter__.return_value.read.return_value = json.dumps(mock_content)
            
            # Download collection
            result = await cloud_storage.download_collection(test_collection)
            
            # Verify download
            mock_s3.download_file.assert_called_once()
            assert result["status"] == "success"
            assert result["downloaded_files"] == 2  # vectors and metadata


@pytest.mark.asyncio
async def test_list_collections(cloud_storage: CloudStorage):
    """Test list collections."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "test/collection1/vectors.json"},
            {"Key": "test/collection1/metadata.json"},
            {"Key": "test/collection2/vectors.json"}
        ]
    }
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        # List collections
        collections = await cloud_storage.list_collections()
        
        # Verify list
        mock_s3.list_objects_v2.assert_called_once()
        assert len(collections) == 2
        assert "collection1" in collections
        assert "collection2" in collections


@pytest.mark.asyncio
async def test_delete_collection(cloud_storage: CloudStorage):
    """Test delete collection."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.delete_objects.return_value = {
        "Deleted": [
            {"Key": "test/collection1/vectors.json"},
            {"Key": "test/collection1/metadata.json"}
        ]
    }
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        # Delete collection
        result = await cloud_storage.delete_collection("collection1")
        
        # Verify delete
        mock_s3.delete_objects.assert_called_once()
        assert result["status"] == "success"
        assert result["deleted_files"] == 2


@pytest.mark.asyncio
async def test_sync_collections(
    cloud_storage: CloudStorage,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test sync collections."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.list_objects_v2.return_value = {
        "Contents": [
            {"Key": f"test/{test_collection}/vectors.json"},
            {"Key": f"test/{test_collection}/metadata.json"}
        ]
    }
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        with patch("aiofiles.open", AsyncMock()) as mock_file:
            mock_file.return_value.__aenter__.return_value.read.return_value = json.dumps(test_collection_data)
            
            # Sync collections
            result = await cloud_storage.sync_collections()
            
            # Verify sync
            assert result["status"] == "success"
            assert result["synced_collections"] == 1


@pytest.mark.asyncio
async def test_error_handling(cloud_storage: CloudStorage):
    """Test error handling."""
    # Mock S3 client
    mock_s3 = AsyncMock()
    mock_s3.list_objects_v2.side_effect = Exception("S3 error")
    
    with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
        mock_session.return_value.client.return_value = mock_s3
        
        # Test error handling
        with pytest.raises(CloudStorageError):
            await cloud_storage.list_collections()


@pytest.mark.asyncio
async def test_context_manager(
    milvus_config: MilvusConfig,
    storage_config: CloudStorageConfig
):
    """Test context manager."""
    async with CloudStorage(milvus_config, storage_config) as storage:
        # Mock S3 client
        mock_s3 = AsyncMock()
        mock_s3.list_objects_v2.return_value = {"Contents": []}
        
        with patch("aioboto3.Session", return_value=AsyncMock()) as mock_session:
            mock_session.return_value.client.return_value = mock_s3
            
            # Test context manager
            collections = await storage.list_collections()
            assert collections == [] 