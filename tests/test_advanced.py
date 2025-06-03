"""
Unit tests for advanced Milvus features.
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
import json
import time

from ai_prishtina_milvus_client import (
    MilvusConfig,
    AdvancedMilvusClient,
    PartitionConfig,
    HybridQueryConfig,
    StreamConfig,
    KafkaStreamProcessor,
    StreamMessage
)


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
        nlist=1024
    )


@pytest.fixture
def client(config):
    """Create advanced client with mocked collection."""
    client = AdvancedMilvusClient(config)
    client.collection = MagicMock()
    return client


@pytest.fixture
def sample_data():
    """Generate sample vectors and metadata."""
    np.random.seed(42)
    vectors = np.random.rand(10, 128).tolist()
    metadata = [
        {
            "id": i,
            "category": np.random.choice(["A", "B", "C"]),
            "score": np.random.uniform(0, 1)
        }
        for i in range(10)
    ]
    return vectors, metadata


def test_create_partition(client):
    """Test partition creation."""
    partition_config = PartitionConfig(
        partition_name="test_partition",
        description="Test partition",
        tags=["test"]
    )
    client.create_partition(partition_config)
    client.collection.create_partition.assert_called_once_with(
        partition_name="test_partition",
        description="Test partition",
        tags=["test"]
    )


def test_drop_partition(client):
    """Test partition deletion."""
    client.drop_partition("test_partition")
    client.collection.drop_partition.assert_called_once_with("test_partition")


def test_list_partitions(client):
    """Test listing partitions."""
    # Mock partition objects
    mock_partitions = [
        MagicMock(
            name="p1",
            description="Partition 1",
            tags=["tag1"],
            num_entities=100
        ),
        MagicMock(
            name="p2",
            description="Partition 2",
            tags=["tag2"],
            num_entities=200
        )
    ]
    client.collection.partitions = mock_partitions
    
    partitions = client.list_partitions()
    assert len(partitions) == 2
    assert partitions[0]["name"] == "p1"
    assert partitions[1]["name"] == "p2"


def test_hybrid_search(client, sample_data):
    """Test hybrid search with filters."""
    vectors, metadata = sample_data
    query_config = HybridQueryConfig(
        vector_field="vector",
        scalar_fields=["category", "score"],
        metric_type="L2",
        top_k=5
    )
    
    # Mock search results
    mock_hits = [
        MagicMock(
            id=i,
            distance=0.1 * i,
            entity={"category": "A", "score": 0.5}
        )
        for i in range(5)
    ]
    client.collection.search.return_value = [[mock_hits]]
    
    results = client.hybrid_search(
        [vectors[0]],
        query_config,
        partition_names=["category_A"],
        category="A",
        score=0.5
    )
    
    assert len(results) == 1
    assert len(results[0]) == 5
    assert results[0][0]["category"] == "A"
    assert results[0][0]["score"] == 0.5


def test_create_index(client):
    """Test index creation."""
    client.create_index(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    client.collection.create_index.assert_called_once_with(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )


def test_get_index_info(client):
    """Test getting index information."""
    mock_index = MagicMock(
        field_name="vector",
        index_type="IVF_FLAT",
        metric_type="L2",
        params={"nlist": 1024}
    )
    client.collection.index.return_value = mock_index
    
    info = client.get_index_info("vector")
    assert info["field_name"] == "vector"
    assert info["index_type"] == "IVF_FLAT"
    assert info["metric_type"] == "L2"
    assert info["params"] == {"nlist": 1024}


def test_load_release_partition(client):
    """Test loading and releasing partitions."""
    client.load_partition("test_partition")
    client.collection.load_partition.assert_called_once_with("test_partition")
    
    client.release_partition("test_partition")
    client.collection.release_partition.assert_called_once_with("test_partition")


def test_get_partition_stats(client):
    """Test getting partition statistics."""
    mock_partition = MagicMock(
        name="test_partition",
        description="Test partition",
        tags=["test"],
        num_entities=100,
        is_loaded=True
    )
    client.collection.partition.return_value = mock_partition
    
    stats = client.get_partition_stats("test_partition")
    assert stats["name"] == "test_partition"
    assert stats["description"] == "Test partition"
    assert stats["tags"] == ["test"]
    assert stats["num_entities"] == 100
    assert stats["is_loaded"] is True


def test_compact_collection(client):
    """Test collection compaction."""
    client.compact()
    client.collection.compact.assert_called_once()


def test_get_compaction_state(client):
    """Test getting compaction state."""
    mock_state = MagicMock(
        state="completed",
        executing_plans=[],
        timeout_plans=[],
        completed_plans=["plan1"]
    )
    client.collection.get_compaction_state.return_value = mock_state
    
    state = client.get_compaction_state()
    assert state["state"] == "completed"
    assert state["completed_plans"] == ["plan1"]


@pytest.fixture
def stream_config():
    """Create stream configuration."""
    return StreamConfig(
        bootstrap_servers="localhost:9092",
        group_id="test_group",
        topics=["test_topic"],
        batch_size=10,
        num_workers=2
    )


@patch("confluent_kafka.Consumer")
@patch("confluent_kafka.Producer")
def test_kafka_stream_processor(mock_producer, mock_consumer, config, stream_config):
    """Test Kafka stream processor."""
    # Create a mock client
    mock_client = MagicMock()
    mock_client.insert = MagicMock()
    
    # Create processor with mock client
    processor = KafkaStreamProcessor(config, stream_config, client=mock_client)
    
    # Test message production
    message = StreamMessage(
        vectors=[[1.0, 2.0, 3.0]],
        metadata=[{"id": 1}],
        operation="insert",
        collection="test_collection"
    )
    processor.produce_message("test_topic", message)
    mock_producer.return_value.produce.assert_called_once()
    
    # Test message processing
    mock_msg = MagicMock()
    mock_msg.value.return_value = json.dumps(message.__dict__).encode("utf-8")
    mock_consumer.return_value.poll.return_value = mock_msg
    
    processor.start()
    time.sleep(0.1)  # Allow time for processing
    processor.stop()
    
    mock_consumer.return_value.close.assert_called_once() 