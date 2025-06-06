"""
Advanced integration test: end-to-end file ingestion, search, and validation using MilvusClient.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import asyncio
from typing import List, Dict, Any
from unittest.mock import AsyncMock, patch

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient
from ai_prishtina_milvus_client.client import AsyncMilvusClient
from ai_prishtina_milvus_client.exceptions import MilvusError


def create_temp_csv(vectors, metadata):
    df = pd.DataFrame({
        "vector": [str(v) for v in vectors],
        "category": [m["category"] for m in metadata],
        "score": [m["score"] for m in metadata],
        "tags": [str(m["tags"]) for m in metadata],
    })
    temp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(temp.name, index=False)
    return temp.name


def create_temp_config(csv_path, collection_name="integration_test_collection"):
    config = {
        "vector_field": "vector",
        "metadata_fields": ["category", "score", "tags"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    yaml.dump(config, temp)
    return temp.name


def create_milvus_config(collection_name="integration_test_collection", dim=128, metadata_fields=None):
    return MilvusConfig(
        host="localhost",
        port=19530,
        collection_name=collection_name,
        dim=dim,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
        metadata_fields=metadata_fields,
    )


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
async def test_collection_operations(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test collection operations."""
    # Create collection
    await milvus_client.create_collection(
        collection_name=test_collection,
        dimension=3,
        index_type="IVF_FLAT",
        metric_type="L2"
    )
    
    # Insert vectors
    inserted_ids = await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
    assert len(inserted_ids) == len(test_collection_data["vectors"])
    
    # Query vectors
    results = await milvus_client.query(
        collection_name=test_collection,
        expr="id in [0, 1, 2]",
        output_fields=["id"]
    )
    
    assert len(results) == 3
    
    # Search vectors
    search_results = await milvus_client.search(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"][:2],
        limit=5
    )
    
    assert len(search_results) == 2
    assert len(search_results[0]) == 5
    
    # Delete vectors
    deleted_count = await milvus_client.delete(
        collection_name=test_collection,
        expr="id in [0, 1, 2]"
    )
    
    assert deleted_count == 3
    
    # Drop collection
    await milvus_client.drop_collection(test_collection)


@pytest.mark.asyncio
async def test_partition_operations(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test partition operations."""
    # Create collection
    await milvus_client.create_collection(
        collection_name=test_collection,
        dimension=3,
        index_type="IVF_FLAT",
        metric_type="L2"
    )
    
    # Create partition
    partition_name = "test_partition"
    await milvus_client.create_partition(
        collection_name=test_collection,
        partition_name=partition_name
    )
    
    # Insert vectors into partition
    inserted_ids = await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"],
        partition_name=partition_name
    )
    
    assert len(inserted_ids) == len(test_collection_data["vectors"])
    
    # Query partition
    results = await milvus_client.query(
        collection_name=test_collection,
        expr="id in [0, 1, 2]",
        output_fields=["id"],
        partition_names=[partition_name]
    )
    
    assert len(results) == 3
    
    # Drop partition
    await milvus_client.drop_partition(
        collection_name=test_collection,
        partition_name=partition_name
    )
    
    # Drop collection
    await milvus_client.drop_collection(test_collection)


@pytest.mark.asyncio
async def test_index_operations(
    milvus_client: AsyncMilvusClient,
    test_collection: str,
    test_collection_data: Dict[str, Any]
):
    """Test index operations."""
    # Create collection
    await milvus_client.create_collection(
        collection_name=test_collection,
        dimension=3,
        index_type="IVF_FLAT",
        metric_type="L2"
    )
    
    # Insert vectors
    await milvus_client.insert_vectors(
        collection_name=test_collection,
        vectors=test_collection_data["vectors"],
        metadata=test_collection_data["metadata"]
    )
    
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
    
    assert index_info["index_type"] == "IVF_FLAT"
    assert index_info["metric_type"] == "L2"
    
    # Drop index
    await milvus_client.drop_index(
        collection_name=test_collection,
        field_name="vector"
    )
    
    # Drop collection
    await milvus_client.drop_collection(test_collection)


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


def test_end_to_end_file_ingestion_and_search():
    # Generate test data
    vectors = np.random.rand(50, 128).tolist()
    metadata = [
        {
            "category": np.random.choice(["A", "B", "C"]),
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
        }
        for _ in range(50)
    ]
    csv_path = create_temp_csv(vectors, metadata)
    config_path = create_temp_config(csv_path)
    milvus_config = create_milvus_config(metadata_fields=[
        {"name": "category", "type": "str"},
        {"name": "score", "type": "float"},
        {"name": "tags", "type": "str"},
    ])

    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 50
        query_vector = vectors[0]
        results = client.search([query_vector], top_k=5)
        assert len(results) == 1
        assert len(results[0]) == 5
        for r in results[0]:
            assert "id" in r and "distance" in r
        client.drop_collection()

    # Clean up
    os.unlink(csv_path)
    os.unlink(config_path)


def test_hybrid_search_with_metadata_filter():
    # Generate test data
    vectors = np.random.rand(50, 128).tolist()
    categories = [np.random.choice(["A", "B", "C"]) for _ in range(50)]
    metadata = [
        {
            "category": cat,
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
        }
        for cat in categories
    ]
    csv_path = create_temp_csv(vectors, metadata)
    config_path = create_temp_config(csv_path, collection_name="hybrid_test_collection")
    milvus_config = create_milvus_config(collection_name="hybrid_test_collection", metadata_fields=[
        {"name": "category", "type": "str"},
        {"name": "score", "type": "float"},
        {"name": "tags", "type": "str"},
    ])

    # Ingest and hybrid search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 50
        query_vector = vectors[0]
        target_category = metadata[0]["category"]
        # Hybrid search: vector + metadata filter
        results = client.search(
            [query_vector],
            top_k=5,
            filter=f"category == '{target_category}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["category"] == target_category
        client.drop_collection()

    # Clean up
    os.unlink(csv_path)
    os.unlink(config_path)


def test_multimodal_image_text_ingestion_and_search():
    # Generate test data
    image_vectors = np.random.rand(30, 256).tolist()  # e.g., image embeddings
    captions = [f"A photo of a {obj}" for obj in np.random.choice(["cat", "dog", "car", "tree"], size=30)]
    metadata = [
        {
            "caption": cap,
            "source": np.random.choice(["web", "user", "dataset"]),
        }
        for cap in captions
    ]
    # Save as CSV
    df = pd.DataFrame({
        "vector": [str(v) for v in image_vectors],
        "caption": [m["caption"] for m in metadata],
        "source": [m["source"] for m in metadata],
    })
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df.to_csv(csv_path, index=False)
    # Config
    config = {
        "vector_field": "vector",
        "metadata_fields": ["caption", "source"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    config_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="multimodal_image_text",
        dim=256,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
        metadata_fields=[
            {"name": "caption", "type": "str"},
            {"name": "source", "type": "str"},
        ],
    )
    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 30
        query_vector = image_vectors[0]
        target_caption = metadata[0]["caption"]
        # Search by vector and filter by caption
        results = client.search(
            [query_vector],
            top_k=3,
            filter=f"caption == '{target_caption}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["caption"] == target_caption
        client.drop_collection()
    os.unlink(csv_path)
    os.unlink(config_path)


def test_multimodal_audio_text_ingestion_and_search():
    # Generate test data
    audio_vectors = np.random.rand(20, 128).tolist()  # e.g., audio embeddings
    transcripts = [
        f"This is a recording of a {obj}."
        for obj in np.random.choice(["meeting", "lecture", "podcast", "interview"], size=20)
    ]
    metadata = [
        {
            "transcript": t,
            "duration": float(np.random.randint(30, 300)),  # seconds
        }
        for t in transcripts
    ]
    # Save as CSV
    df = pd.DataFrame({
        "vector": [str(v) for v in audio_vectors],
        "transcript": [m["transcript"] for m in metadata],
        "duration": [m["duration"] for m in metadata],
    })
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df.to_csv(csv_path, index=False)
    # Config
    config = {
        "vector_field": "vector",
        "metadata_fields": ["transcript", "duration"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    config_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="multimodal_audio_text",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
        metadata_fields=[
            {"name": "transcript", "type": "str"},
            {"name": "duration", "type": "float"},
        ],
    )
    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 20
        query_vector = audio_vectors[0]
        target_transcript = metadata[0]["transcript"]
        # Search by vector and filter by transcript
        results = client.search(
            [query_vector],
            top_k=2,
            filter=f"transcript == '{target_transcript}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["transcript"] == target_transcript
        client.drop_collection()
    os.unlink(csv_path)
    os.unlink(config_path) 