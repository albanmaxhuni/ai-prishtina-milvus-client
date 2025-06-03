# AI Prishtina Milvus Client

![AI Prishtina Logo](assets/png/ai-prishtina.jpeg)

[![PyPI version](https://badge.fury.io/py/ai-prishtina-milvus-client.svg)](https://badge.fury.io/py/ai-prishtina-milvus-client)
[![Python Versions](https://img.shields.io/pypi/pyversions/ai-prishtina-milvus-client.svg)](https://pypi.org/project/ai-prishtina-milvus-client/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/ai-prishtina-milvus-client/badge/?version=latest)](https://ai-prishtina-milvus-client.readthedocs.io/en/latest/?badge=latest)

A high-performance Python client for Milvus vector database, designed specifically for AI applications with a focus on scalability, reliability, and ease of use.

## Author
**Alban Maxhuni, PhD**  
Email: [alban.q.maxhuni@gmail.com](mailto:alban.q.maxhuni@gmail.com) | [info@albanmaxhuni.com](mailto:info@albanmaxhuni.com)

## Key Features

### Core Functionality
- **Vector Operations**: Efficient insert, search, and management of high-dimensional vectors
- **Metadata Support**: Flexible metadata storage and querying alongside vectors
- **Type Safety**: Comprehensive type hints and runtime validation using Pydantic
- **Error Handling**: Robust error recovery and graceful degradation

### AI Integration
- **Multiple API Providers**: Native support for:
  - OpenAI GPT models
  - Google Vertex AI
  - AWS Bedrock
  - Custom model endpoints
- **Embedding Generation**: Automatic vector generation from text, images, and audio
- **Hybrid Search**: Combine vector similarity with metadata filtering

### Advanced Features
- **Streaming Support**: Real-time vector ingestion through Kafka
- **Partition Management**: Efficient data organization and querying
- **Advanced Indexing**: Support for multiple index types (IVF_FLAT, HNSW, etc.)
- **Collection Compaction**: Automatic data optimization
- **Batch Operations**: Efficient bulk data processing
- **Cloud Storage Integration**: Native support for S3, GCP, and Azure

### Enterprise Features
- **Security**: Role-based access control and data encryption
- **Monitoring**: Comprehensive metrics collection and monitoring
- **Backup & Recovery**: Automated backup and disaster recovery
- **Performance Optimization**: Query optimization and caching

## Installation

```bash
# Basic installation
pip install ai-prishtina-milvus-client

# With optional dependencies
pip install ai-prishtina-milvus-client[all]
```

## Quick Start

```python
from ai_prishtina_milvus_client import MilvusClient, MilvusConfig

# Configure client
config = MilvusConfig(
    host="localhost",
    port=19530,
    collection_name="my_collection",
    dimension=128,
    index_type="IVF_FLAT",
    metric_type="L2"
)

# Initialize client
client = MilvusClient(config)

# Insert vectors with metadata
vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
metadata = [{"id": 1, "category": "A"}, {"id": 2, "category": "B"}]
client.insert(vectors, metadata)

# Search vectors
results = client.search(
    query_vectors=[query_vector],
    top_k=5,
    filter="category == 'A'"
)
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
    batch_size=100,
    num_workers=4
)

# Initialize processor
processor = KafkaStreamProcessor(milvus_config, stream_config)
processor.start()
```

### Hybrid Search with Metadata

```python
from ai_prishtina_milvus_client import HybridQueryConfig

# Configure hybrid search
query_config = HybridQueryConfig(
    vector_field="vector",
    scalar_fields=["category", "score"],
    metric_type="L2",
    top_k=5
)

# Perform search with filters
results = client.hybrid_search(
    query_vectors=[query_vector],
    query_config=query_config,
    filter="category == 'A' and score > 0.5"
)
```

### Cloud Storage Integration

```python
from ai_prishtina_milvus_client import CloudStorageConfig

# Configure cloud storage
storage_config = CloudStorageConfig(
    provider="s3",
    bucket="my-bucket",
    region="us-west-2"
)

# Import data from cloud storage
client.import_from_cloud(
    storage_config,
    file_path="vectors.parquet",
    batch_size=1000
)
```

## Documentation

- [Basic Usage](docs/basic_usage.md)
- [Advanced Features](docs/advanced_features.md)
- [API Reference](docs/api_reference.md)
- [Examples](examples/)
- [Performance Tuning](docs/performance.md)
- [Security Guide](docs/security.md)

## Development

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- Milvus 2.3.3+

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/albanmaxhuni/ai-prishtina-milvus-client.git
cd ai-prishtina-milvus-client

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Start Milvus
docker-compose up -d

# Run tests
pytest

# Run linting
flake8
black .
isort .
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Milvus](https://milvus.io/) - Vector database
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation
- [Confluent Kafka](https://docs.confluent.io/) - Streaming support
- [Apache Arrow](https://arrow.apache.org/) - Data processing
- [NumPy](https://numpy.org/) - Numerical computing

## Support

- Documentation: [Read the Docs](https://ai-prishtina-milvus-client.readthedocs.io/)
- Issues: [GitHub Issues](https://github.com/albanmaxhuni/ai-prishtina-milvus-client/issues)
- Email: [alban.q.maxhuni@gmail.com](mailto:alban.q.maxhuni@gmail.com)