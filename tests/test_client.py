"""
Tests for the Milvus client.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

import numpy as np
import pytest
from pymilvus import utility
import asyncio
from typing import List, Dict, Any
import json

from ai_prishtina_milvus_client import MilvusClient
from ai_prishtina_milvus_client.config import MilvusConfig
from ai_prishtina_milvus_client.exceptions import (
    CollectionError,
    ConnectionError,
    InsertError,
    SearchError,
    MilvusError,
)
from ai_prishtina_milvus_client.client import AsyncMilvusClient


@pytest.fixture
def config_file():
    """Create a temporary config file for testing."""
    config = {
        "milvus": {
            "host": "localhost",
            "port": 19530,
            "db_name": "default",
            "collection_name": "test_collection",
            "dim": 128,
            "index_type": "IVF_FLAT",
            "metric_type": "L2",
            "nlist": 1024,
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml
        yaml.dump(config, f)
        return f.name


@pytest.fixture
def mock_milvus():
    """Create mock Milvus connection."""
    with patch("pymilvus.connections.connect") as mock_connect, \
         patch("pymilvus.connections.disconnect") as mock_disconnect, \
         patch("pymilvus.utility.has_collection") as mock_has_collection, \
         patch("pymilvus.utility.drop_collection") as mock_drop_collection:
        yield {
            "connect": mock_connect,
            "disconnect": mock_disconnect,
            "has_collection": mock_has_collection,
            "drop_collection": mock_drop_collection
        }


@pytest.fixture
def client(config_file, mock_milvus):
    """Create a Milvus client instance for testing."""
    client = MilvusClient(config_file)
    yield client
    client.close()
    if mock_milvus["has_collection"].return_value:
        mock_milvus["drop_collection"]("test_collection")
    os.unlink(config_file)


def test_create_collection(client, mock_milvus):
    """Test collection creation."""
    mock_milvus["has_collection"].return_value = False
    client.create_collection()
    assert mock_milvus["has_collection"].called


def test_insert_and_search(client, mock_milvus):
    """Test vector insertion and search."""
    # Create collection
    mock_milvus["has_collection"].return_value = False
    client.create_collection()
    
    # Generate test vectors
    vectors = np.random.rand(100, 128).tolist()
    
    # Insert vectors
    client.insert(vectors)
    
    # Search for similar vectors
    query_vector = vectors[0]
    results = client.search([query_vector], top_k=5)
    
    assert len(results) == 1  # One query vector
    assert len(results[0]) == 5  # Top 5 results
    assert all(isinstance(r["id"], int) for r in results[0])
    assert all(isinstance(r["distance"], float) for r in results[0])


def test_delete(client, mock_milvus):
    """Test vector deletion."""
    # Create collection
    mock_milvus["has_collection"].return_value = False
    client.create_collection()
    
    # Insert vectors
    vectors = np.random.rand(10, 128).tolist()
    client.insert(vectors)
    
    # Delete vectors
    client.delete("id in [1, 2, 3]")
    
    # Verify deletion
    results = client.search([vectors[0]], top_k=10)
    assert len(results[0]) < 10


def test_invalid_config():
    """Test handling of invalid configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        f.write("invalid: yaml")
        f.flush()
        with pytest.raises(ValueError):
            MilvusClient(f.name)


def test_connection_error(mock_milvus):
    """Test handling of connection errors."""
    mock_milvus["connect"].side_effect = Exception("Connection failed")
    config = {
        "milvus": {
            "host": "invalid_host",
            "port": 19530,
            "collection_name": "test_collection",
            "dim": 128,
        }
    }
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        import yaml
        yaml.dump(config, f)
        f.flush()
        with pytest.raises(ConnectionError):
            MilvusClient(f.name)


@pytest.fixture
def config():
    """Create Milvus configuration."""
    return MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="test_collection",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return {
        "vectors": np.random.rand(100, 128).tolist(),
        "metadata": [
            {
                "category": np.random.choice(["A", "B", "C"]),
                "score": float(np.random.randint(0, 100)),
                "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
            }
            for _ in range(100)
        ],
    }


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_create_collection(mock_collection, mock_connect, config):
    """Test creating a collection."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Create client
    with MilvusClient(config) as client:
        # Create collection
        client.create_collection()
        
        # Verify collection creation
        mock_collection.assert_called_once()
        mock_coll.create_index.assert_called_once()


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_insert_vectors(mock_collection, mock_connect, config, sample_data):
    """Test inserting vectors."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Create client
    with MilvusClient(config) as client:
        # Insert vectors
        client.insert(sample_data["vectors"], sample_data["metadata"])
        
        # Verify insertion
        mock_coll.insert.assert_called_once()


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_search_vectors(mock_collection, mock_connect, config):
    """Test searching vectors."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Mock search results
    mock_results = [
        {
            "id": 1,
            "distance": 0.1,
            "category": "A",
            "score": 0.8,
            "tags": ["tag1", "tag2"],
        },
        {
            "id": 2,
            "distance": 0.2,
            "category": "B",
            "score": 0.9,
            "tags": ["tag2", "tag3"],
        },
    ]
    mock_coll.search.return_value = [mock_results]
    
    # Create client
    with MilvusClient(config) as client:
        # Search vectors
        query_vector = np.random.rand(128).tolist()
        results = client.search([query_vector], top_k=2)
        
        # Verify search
        assert len(results) == 1
        assert len(results[0]) == 2
        assert results[0][0]["id"] == 1
        assert results[0][1]["id"] == 2


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_get_collection_stats(mock_collection, mock_connect, config):
    """Test getting collection statistics."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Mock statistics
    mock_stats = {
        "row_count": 100,
        "partitions": [
            {"tag": "default", "row_count": 100},
        ],
    }
    mock_coll.get_statistics.return_value = mock_stats
    
    # Create client
    with MilvusClient(config) as client:
        # Get statistics
        stats = client.get_collection_stats()
        
        # Verify statistics
        assert stats["row_count"] == 100
        assert len(stats["partitions"]) == 1


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_drop_collection(mock_collection, mock_connect, config):
    """Test dropping a collection."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Create client
    with MilvusClient(config) as client:
        # Drop collection
        client.drop_collection()
        
        # Verify collection drop
        mock_coll.drop.assert_called_once()


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_insert_from_cloud(mock_collection, mock_connect, config, sample_data):
    """Test inserting vectors from cloud storage."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Create temporary cloud config
    cloud_config = {
        "service": "s3",
        "bucket_name": "test-bucket",
        "region_name": "us-east-1",
        "aws_access_key_id": "test-key",
        "aws_secret_access_key": "test-secret",
    }
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_file.write(str(cloud_config).encode())
        cloud_config_path = temp_file.name
    
    # Create client
    with MilvusClient(config) as client:
        # Mock cloud storage download
        with patch("boto3.client") as mock_boto3:
            mock_s3 = MagicMock()
            mock_boto3.return_value = mock_s3
            
            # Create temporary data file
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as data_file:
                data_file.write(str(sample_data).encode())
                data_path = data_file.name
            
            # Mock S3 download
            mock_s3.download_file.return_value = None
            
            # Insert from cloud
            client.insert_from_cloud(cloud_config_path, "test-file.json")
            
            # Verify insertion
            mock_coll.insert.assert_called_once()
            
            # Clean up
            os.unlink(data_path)
    
    # Clean up
    os.unlink(cloud_config_path)


@patch("pymilvus.connections.connect")
@patch("pymilvus.Collection")
def test_insert_from_api(mock_collection, mock_connect, config):
    """Test inserting vectors from API."""
    # Mock collection
    mock_coll = MagicMock()
    mock_collection.return_value = mock_coll
    
    # Create temporary API config
    api_config = {
        "service": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "model": "text-embedding-ada-002",
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.7,
        },
    }
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_file.write(str(api_config).encode())
        api_config_path = temp_file.name
    
    # Create client
    with MilvusClient(config) as client:
        # Mock API client
        with patch("requests.Session") as mock_session:
            mock_sess = MagicMock()
            mock_session.return_value = mock_sess
            
            # Mock API response
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }
            mock_sess.post.return_value = mock_response
            
            # Insert from API
            client.insert_from_api(api_config_path, "test query")
            
            # Verify insertion
            mock_coll.insert.assert_called_once()
    
    # Clean up
    os.unlink(api_config_path)


def test_invalid_config():
    """Test invalid configuration."""
    # Missing required fields
    config = MilvusConfig(
        host="localhost",
        port=19530,
    )
    
    with pytest.raises(ValueError):
        MilvusClient(config)


def test_invalid_index_type():
    """Test invalid index type."""
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="test_collection",
        dim=128,
        index_type="INVALID",
        metric_type="L2",
        nlist=1024,
    )
    
    with pytest.raises(ValueError):
        MilvusClient(config)


def test_invalid_metric_type():
    """Test invalid metric type."""
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="test_collection",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="INVALID",
        nlist=1024,
    )
    
    with pytest.raises(ValueError):
        MilvusClient(config)


@pytest.fixture
async def milvus_client(milvus_config: MilvusConfig):
    """Create Milvus client."""
    client = AsyncMilvusClient(milvus_config)
    yield client
    await client.cleanup()


@pytest.fixture
def test_collection_data() -> Dict[str, Any]:
    """Generate test collection data."""
    return {
        "vectors": [[0.1, 0.2, 0.3] for _ in range(10)],
        "metadata": [{"id": i} for i in range(10)]
    }


@pytest.mark.asyncio
async def test_create_collection(milvus_client: AsyncMilvusClient):
    """Test create collection."""
    # Create collection
    await milvus_client.create_collection(
        collection_name="test_collection",
        dimension=3,
        index_type="IVF_FLAT",
        metric_type="L2"
    )
    
    # Verify collection exists
    collections = await milvus_client.list_collections()
    assert "test_collection" in collections
    
    # Drop collection
    await milvus_client.drop_collection("test_collection")


@pytest.mark.asyncio
async def test_insert_vectors(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test insert vectors."""
    # Insert vectors
    inserted_ids = await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
    # Verify insertion
    assert len(inserted_ids) == len(test_collection_data["vectors"])
    
    # Query inserted vectors
    results = await milvus_client.query(
        collection_name=test_collection,
        expr="id in [0, 1, 2]",
        output_fields=["id"]
    )
    
    assert len(results) == 3


@pytest.mark.asyncio
async def test_query_vectors(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test query vectors."""
    # Insert vectors
    await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
    # Query vectors
    results = await milvus_client.query(
        collection_name=test_collection,
        expr="id in [0, 1, 2]",
        output_fields=["id", "vector"]
    )
    
    # Verify results
    assert len(results) == 3
    for result in results:
        assert "id" in result
        assert "vector" in result


@pytest.mark.asyncio
async def test_search_vectors(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test search vectors."""
    # Insert vectors
    await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
    # Search vectors
    search_results = await milvus_client.search(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"][:2],
        limit=5
    )
    
    # Verify results
    assert len(search_results) == 2
    assert len(search_results[0]) == 5
    for result in search_results[0]:
        assert "id" in result
        assert "distance" in result


@pytest.mark.asyncio
async def test_delete_vectors(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test delete vectors."""
    # Insert vectors
    await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
    # Delete vectors
    deleted_count = await milvus_client.delete(
        collection_name=test_collection,
        expr="id in [0, 1, 2]"
    )
    
    # Verify deletion
    assert deleted_count == 3
    
    # Query deleted vectors
    results = await milvus_client.query(
        collection_name=test_collection,
        expr="id in [0, 1, 2]",
        output_fields=["id"]
    )
    
    assert len(results) == 0


@pytest.mark.asyncio
async def test_partition_operations(
    milvus_client: AsyncMilvusClient,
    test_collection: str
):
    """Test partition operations."""
    # Create partition
    partition_name = "test_partition"
    await milvus_client.create_partition(
        collection_name=test_collection,
        partition_name=partition_name
    )
    
    # List partitions
    partitions = await milvus_client.list_partitions(test_collection)
    assert partition_name in partitions
    
    # Drop partition
    await milvus_client.drop_partition(
        collection_name=test_collection,
        partition_name=partition_name
    )
    
    # Verify partition is dropped
    partitions = await milvus_client.list_partitions(test_collection)
    assert partition_name not in partitions


@pytest.mark.asyncio
async def test_index_operations(
    milvus_client: AsyncMilvusClient,
    test_collection: str
):
    """Test index operations."""
    # Create index
    await milvus_client.create_index(
        collection_name=test_collection,
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    
    # Describe index
    index_info = await milvus_client.describe_index(
        collection_name=test_collection,
        field_name="vector"
    )
    
    # Verify index
    assert index_info["index_type"] == "IVF_FLAT"
    assert index_info["metric_type"] == "L2"
    
    # Drop index
    await milvus_client.drop_index(
        collection_name=test_collection,
        field_name="vector"
    )


@pytest.mark.asyncio
async def test_error_handling(milvus_client: AsyncMilvusClient):
    """Test error handling."""
    # Test invalid collection
    with pytest.raises(MilvusError):
        await milvus_client.query(
            collection_name="invalid_collection",
            expr="id in [0, 1, 2]"
        )
    
    # Test invalid partition
    with pytest.raises(MilvusError):
        await milvus_client.create_partition(
            collection_name="test_collection",
            partition_name="invalid_partition"
        )
    
    # Test invalid index
    with pytest.raises(MilvusError):
        await milvus_client.create_index(
            collection_name="test_collection",
            field_name="invalid_field",
            index_type="IVF_FLAT",
            metric_type="L2"
        )


@pytest.mark.asyncio
async def test_context_manager(milvus_config: MilvusConfig):
    """Test context manager."""
    async with AsyncMilvusClient(milvus_config) as client:
        # Create collection
        await client.create_collection(
            collection_name="test_collection",
            dimension=3,
            index_type="IVF_FLAT",
            metric_type="L2"
        )
        
        # Drop collection
        await client.drop_collection("test_collection") 