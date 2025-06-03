# Advanced Features

This document describes the advanced features available in the AI Prishtina Milvus Client.

## Table of Contents
- [Partition Management](#partition-management)
- [Hybrid Search](#hybrid-search)
- [Streaming Support](#streaming-support)
- [Advanced Indexing](#advanced-indexing)
- [Collection Management](#collection-management)

## Partition Management

Partitions allow you to organize your vectors into logical groups for better management and querying.

### Creating Partitions

```python
from ai_prishtina_milvus_client import AdvancedMilvusClient, PartitionConfig

# Create a partition
partition_config = PartitionConfig(
    partition_name="category_A",
    description="Category A vectors",
    tags=["A"]
)
client.create_partition(partition_config)
```

### Managing Partitions

```python
# List all partitions
partitions = client.list_partitions()
for partition in partitions:
    print(f"Name: {partition['name']}")
    print(f"Description: {partition['description']}")
    print(f"Tags: {partition['tags']}")
    print(f"Entities: {partition['num_entities']}")

# Load a partition into memory
client.load_partition("category_A")

# Release a partition from memory
client.release_partition("category_A")

# Get partition statistics
stats = client.get_partition_stats("category_A")
print(f"Loaded: {stats['is_loaded']}")
print(f"Entities: {stats['num_entities']}")

# Drop a partition
client.drop_partition("category_A")
```

## Hybrid Search

Hybrid search combines vector similarity search with scalar field filtering.

### Basic Hybrid Search

```python
from ai_prishtina_milvus_client import HybridQueryConfig

# Configure hybrid search
query_config = HybridQueryConfig(
    vector_field="vector",
    scalar_fields=["category", "score"],
    metric_type="L2",
    top_k=5,
    params={"nprobe": 10}
)

# Perform search with filters
results = client.hybrid_search(
    [query_vector],
    query_config,
    partition_names=["category_A"],
    category="A",
    score=0.5
)
```

### Advanced Filtering

```python
# Multiple partition search
results = client.hybrid_search(
    [query_vector],
    query_config,
    partition_names=["category_A", "category_B"],
    category="A"
)

# Complex filters
results = client.hybrid_search(
    [query_vector],
    query_config,
    score=0.5,
    timestamp=1234567890
)
```

## Streaming Support

The client supports real-time vector ingestion through Kafka.

### Configuration

```python
from ai_prishtina_milvus_client import StreamConfig, KafkaStreamProcessor

# Configure streaming
stream_config = StreamConfig(
    bootstrap_servers="localhost:9092",
    group_id="milvus_demo",
    topics=["vector_ingest"],
    batch_size=100,
    num_workers=2
)

# Initialize processor
processor = KafkaStreamProcessor(milvus_config, stream_config)
```

### Producing Messages

```python
# Create a message
message = StreamMessage(
    vectors=[[1.0, 2.0, 3.0]],
    metadata=[{"id": 1, "category": "A"}],
    operation="insert",
    collection="my_collection"
)

# Produce message
processor.produce_message("vector_ingest", message)
```

### Processing Messages

```python
# Start processing in background
import threading
stream_thread = threading.Thread(target=processor.start)
stream_thread.daemon = True
stream_thread.start()

# Stop processing
processor.stop()
```

## Advanced Indexing

The client provides advanced indexing capabilities for optimizing search performance.

### Creating Indexes

```python
# Create an index
client.create_index(
    field_name="vector",
    index_type="IVF_FLAT",
    metric_type="L2",
    params={"nlist": 1024}
)

# Get index information
info = client.get_index_info("vector")
print(f"Type: {info['index_type']}")
print(f"Metric: {info['metric_type']}")
print(f"Parameters: {info['params']}")

# Drop an index
client.drop_index("vector")
```

## Collection Management

Advanced collection management features for maintaining data quality.

### Compaction

```python
# Compact collection to remove deleted entities
client.compact()

# Get compaction state
state = client.get_compaction_state()
print(f"State: {state['state']}")
print(f"Completed plans: {state['completed_plans']}")
```

### Best Practices

1. **Partitioning**:
   - Use partitions to organize data by logical groups
   - Load only necessary partitions into memory
   - Use tags for easy partition identification

2. **Hybrid Search**:
   - Combine vector similarity with scalar filters
   - Use partition names to limit search scope
   - Optimize filter expressions for performance

3. **Streaming**:
   - Configure appropriate batch sizes
   - Use multiple workers for parallel processing
   - Monitor message processing latency

4. **Indexing**:
   - Choose index type based on data size and search requirements
   - Tune index parameters for optimal performance
   - Monitor index statistics

5. **Collection Management**:
   - Regular compaction to maintain performance
   - Monitor collection statistics
   - Use appropriate metric types for your use case 