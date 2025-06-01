# AI Prishtina Milvus Client

![AI Prishtina Logo](assets/png/ai-prishtina.jpeg)

## Author
**Alban Maxhuni, PhD**  
Email: [alban.q.maxhuni@gmail.com](mailto:alban.q.maxhuni@gmail.com) | [info@albanmaxhuni.com](mailto:info@albanmaxhuni.com)


A Python client for Milvus vector database with advanced features for AI applications.

## Features

- **Vector Operations**: Insert, search, and manage vectors with ease
- **Metadata Support**: Store and query metadata alongside vectors
- **Multiple API Providers**: Support for OpenAI, Google Vertex AI, and AWS Bedrock
- **Streaming Support**: Real-time vector ingestion through Kafka
- **Advanced Features**:
  - Partition management
  - Hybrid search
  - Advanced indexing
  - Collection compaction
- **Type Safety**: Full type hints and validation
- **Comprehensive Testing**: Unit and integration tests
- **Documentation**: Detailed guides and examples

## Installation

```bash
pip install ai-prishtina-milvus-client
```

## Quick Start

```python
from ai_prishtina_milvus_client import MilvusClient, MilvusConfig

# Configure client
config = MilvusConfig(
    host="localhost",
    port=19530,
    collection_name="my_collection",
    dimension=128
)

# Initialize client
client = MilvusClient(config)

# Insert vectors
vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
metadata = [{"id": 1}, {"id": 2}]
client.insert_vectors(vectors, metadata)

# Search vectors
results = client.search_vectors([query_vector], top_k=5)
```

## Advanced Usage

### Streaming with Kafka

```python
from ai_prishtina_milvus_client import StreamConfig, KafkaStreamProcessor

# Configure streaming
stream_config = StreamConfig(
    bootstrap_servers="localhost:9092",
    group_id="milvus_demo",
    topics=["vector_ingest"],
    batch_size=100
)

# Initialize processor
processor = KafkaStreamProcessor(milvus_config, stream_config)
processor.start()
```

### Hybrid Search

```python
from ai_prishtina_milvus_client import AdvancedMilvusClient, HybridQueryConfig

# Configure hybrid search
query_config = HybridQueryConfig(
    vector_field="vector",
    scalar_fields=["category", "score"],
    metric_type="L2",
    top_k=5
)

# Perform search with filters
results = client.hybrid_search(
    [query_vector],
    query_config,
    category="A",
    score=0.5
)
```

### Partition Management

```python
from ai_prishtina_milvus_client import PartitionConfig

# Create partition
partition_config = PartitionConfig(
    partition_name="category_A",
    description="Category A vectors",
    tags=["A"]
)
client.create_partition(partition_config)

# List partitions
partitions = client.list_partitions()
```

## Documentation

- [Basic Usage](docs/basic_usage.md)
- [Advanced Features](docs/advanced_features.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/albanmaxhuni/ai-prishtina-milvus-client.git
cd ai-prishtina-milvus-client

# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Milvus](https://milvus.io/) - Vector database
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Confluent Kafka](https://docs.confluent.io/) - Streaming support