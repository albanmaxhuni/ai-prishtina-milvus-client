![AI Prishtina Logo](../assets/png/ai-prishtina.jpeg)

# AI Prishtina Milvus Client Documentation

## Author
**Alban Maxhuni, PhD**  
Email: [alban.q.maxhuni@gmail.com](mailto:alban.q.maxhuni@gmail.com) | [info@albanmaxhuni.com](mailto:info@albanmaxhuni.com)

## Overview

The AI Prishtina Milvus Client is a high-level Python library for interacting with Milvus vector database. It provides a simplified interface for common operations and handles connection management, error handling, and configuration.

## Installation

```bash
pip install ai-prishtina-milvus-client
```

## Configuration

### Milvus Configuration

The library uses YAML configuration files for easy setup. Here's an example configuration:

```yaml
milvus:
  host: localhost
  port: 19530
  user: username
  password: password
  db_name: default
  collection_name: my_collection
  dim: 128
  index_type: IVF_FLAT
  metric_type: L2
  nlist: 1024
```

### Data Source Configuration

The library supports multiple data sources through JSON configuration files. Here's an example for each supported format:

```json
// CSV data source
{
    "type": "csv",
    "path": "data.csv",
    "vector_field": "vector",
    "metadata_fields": ["category", "score", "tags"],
    "batch_size": 1000
}

// JSON data source
{
    "type": "json",
    "path": "data.json",
    "vector_field": "vector",
    "metadata_fields": ["category", "score", "tags"],
    "batch_size": 1000
}

// NumPy data source
{
    "type": "numpy",
    "path": "data.npz",
    "vector_field": "vector",
    "metadata_fields": ["category", "score", "tags"],
    "batch_size": 1000
}
```

### Cloud Storage Configuration

The library supports multiple cloud storage providers through JSON configuration files:

```json
// AWS S3 configuration
{
    "provider": "aws",
    "bucket": "my-bucket",
    "prefix": "vectors/",
    "region": "us-west-2",
    "credentials": {
        "access_key_id": "YOUR_ACCESS_KEY",
        "secret_access_key": "YOUR_SECRET_KEY"
    }
}

// Google Cloud Storage configuration
{
    "provider": "gcp",
    "bucket": "my-bucket",
    "prefix": "vectors/",
    "credentials": {
        "type": "service_account",
        "project_id": "YOUR_PROJECT_ID",
        "private_key_id": "YOUR_PRIVATE_KEY_ID",
        "private_key": "YOUR_PRIVATE_KEY",
        "client_email": "YOUR_CLIENT_EMAIL",
        "client_id": "YOUR_CLIENT_ID"
    }
}

// Azure Blob Storage configuration
{
    "provider": "azure",
    "bucket": "my-container",
    "prefix": "vectors/",
    "credentials": {
        "account_name": "YOUR_ACCOUNT_NAME",
        "account_key": "YOUR_ACCOUNT_KEY"
    }
}
```

### API Configuration

The library supports multiple API services through JSON configuration files:

```json
// OpenAI configuration
{
    "service": "openai",
    "base_url": "https://api.openai.com",
    "api_key": "YOUR_API_KEY",
    "timeout": 30
}

// HuggingFace configuration
{
    "service": "huggingface",
    "base_url": "https://api-inference.huggingface.co",
    "api_key": "YOUR_API_KEY",
    "timeout": 30
}

// Cohere configuration
{
    "service": "cohere",
    "base_url": "https://api.cohere.ai",
    "api_key": "YOUR_API_KEY",
    "timeout": 30
}
```

### API Client Architecture

The library provides a flexible and extensible API client architecture for working with different embedding services:

#### APIConfig

The `APIConfig` class defines the configuration for API clients:

```python
class APIConfig:
    service: str          # API service name (e.g., "openai", "huggingface", "cohere")
    base_url: str         # Base URL for the API
    api_key: str          # API key for authentication (optional)
    headers: dict         # Additional headers (optional)
    timeout: int          # Request timeout in seconds (default: 30)
    model: str           # Model name for the service (optional)
    parameters: dict     # Additional model parameters (optional)
```

#### APIClient

The `APIClient` is an abstract base class that provides common functionality for all API clients:

```python
class APIClient:
    def __init__(self, config: APIConfig):
        # Initialize with configuration
        # Sets up session with authentication and headers
        pass

    def get_vectors(self, query: str, **kwargs) -> List[List[float]]:
        # Abstract method to be implemented by specific clients
        pass
```

#### APIClientFactory

The `APIClientFactory` creates appropriate API client instances based on the service type:

```python
class APIClientFactory:
    _clients = {
        "huggingface": HuggingFaceClient,
        "openai": OpenAIClient,
        "cohere": CohereClient,
    }
    
    @classmethod
    def create(cls, config: APIConfig) -> APIClient:
        # Creates and returns appropriate client instance
        pass
```

#### Usage Example

```python
# Load API configuration
config = APIConfig(
    service="openai",
    base_url="https://api.openai.com",
    api_key="YOUR_API_KEY",
    model="text-embedding-ada-002",
    parameters={"temperature": 0.7}
)

# Create client using factory
client = APIClientFactory.create(config)

# Get vectors
vectors = client.get_vectors("Sample text", temperature=0.8)
```

#### Environment Variables

The library supports loading API keys from environment variables:

```python
# If API key is not in config, it will be loaded from environment
os.environ["API_KEY"] = "your-api-key"
client = load_api_client("config.yaml")
```

### Configuration Parameters

#### Milvus Parameters
- `host`: Milvus server host (default: "localhost")
- `port`: Milvus server port (default: 19530)
- `user`: Milvus username (optional)
- `password`: Milvus password (optional)
- `db_name`: Database name (default: "default")
- `collection_name`: Collection name (required)
- `dim`: Vector dimension (required)
- `index_type`: Index type (default: "IVF_FLAT")
- `metric_type`: Distance metric type (default: "L2")
- `nlist`: Number of clusters for IVF index (default: 1024)

#### Data Source Parameters
- `type`: Data source type ("csv", "json", or "numpy")
- `path`: Path to the data file
- `vector_field`: Name of the field containing vectors
- `metadata_fields`: Optional list of metadata field names
- `batch_size`: Number of vectors to process at once (default: 1000)

#### Cloud Storage Parameters
- `provider`: Cloud provider ("aws", "gcp", or "azure")
- `bucket`: Bucket/container name
- `prefix`: Optional path prefix in bucket
- `region`: Optional region for the bucket
- `credentials`: Provider-specific credentials

#### API Parameters
- `service`: API service name ("openai", "huggingface", or "cohere")
- `base_url`: Base URL for the API
- `api_key`: API key for authentication
- `headers`: Optional additional headers
- `timeout`: Request timeout in seconds (default: 30)

## Basic Usage

### Initialization

```python
from ai_prishtina_milvus_client import MilvusClient

# Using regular initialization
client = MilvusClient(config_path="config.yaml")

# Using context manager (recommended)
with MilvusClient(config_path="config.yaml") as client:
    # Use client here
    pass  # Connection is automatically closed
```

### Creating a Collection

```python
client.create_collection()
```

### Inserting Vectors

```python
# Simple vector insertion
vectors = [[1.0] * 128] * 100
client.insert(vectors)

# Insertion with metadata
metadata = [{"category": "A", "score": 0.8} for _ in range(100)]
client.insert(vectors, metadata)

# Insertion from data source
client.insert_from_source("data_source.json")

# Insertion from cloud storage
client.insert_from_cloud("s3_config.json", "vectors/data.csv")

# Insertion from API
client.insert_from_api(
    "openai_config.json",
    "This is a sample text for embedding",
    model="text-embedding-ada-002"
)
```

### Searching Vectors

```python
# Basic search
results = client.search(query_vectors=[[1.0] * 128], top_k=10)

# Search with custom parameters
search_params = {
    "metric_type": "L2",
    "params": {"nprobe": 20}
}
results = client.search(
    query_vectors=[[1.0] * 128],
    top_k=10,
    search_params=search_params
)
```

### Deleting Vectors

```python
# Delete by expression
client.delete("id in [1, 2, 3]")
client.delete("category == 'A'")
```

### Collection Management

```python
# Get collection statistics
stats = client.get_collection_stats()
print(f"Collection statistics: {stats}")

# List all collections
collections = client.list_collections()
print(f"Available collections: {collections}")

# Drop collection
client.drop_collection()
```

### Cleanup

```python
# Manual cleanup
client.close()

# Or use context manager (recommended)
with MilvusClient(config_path="config.yaml") as client:
    # Use client here
    pass  # Connection is automatically closed
```

## Advanced Usage

### Working with Data Sources

The library supports multiple data source formats:

1. **CSV Files**
   ```python
   # CSV file format
   vector,category,score,tags
   "[0.1, 0.2, ...]","A",0.8,"['tag1', 'tag2']"
   ```

2. **JSON Files**
   ```python
   # JSON file format
   [
       {
           "vector": [0.1, 0.2, ...],
           "category": "A",
           "score": 0.8,
           "tags": ["tag1", "tag2"]
       }
   ]
   ```

3. **NumPy Files**
   ```python
   # NumPy file format (.npz)
   import numpy as np
   np.savez(
       "data.npz",
       vector=np.array([[0.1, 0.2, ...]]),
       category=np.array(["A"]),
       score=np.array([0.8]),
       tags=np.array([["tag1", "tag2"]])
   )
   ```

### Working with Cloud Storage

The library supports multiple cloud storage providers:

1. **AWS S3**
   - Supports IAM roles and access keys
   - Automatic region detection
   - Prefix-based file organization

2. **Google Cloud Storage**
   - Supports service account authentication
   - Automatic project detection
   - Prefix-based file organization

3. **Azure Blob Storage**
   - Supports account key and managed identity
   - Automatic endpoint detection
   - Prefix-based file organization

### Working with APIs

The library supports multiple API services:

1. **OpenAI**
   - Text embeddings with various models
   - Metadata generation with GPT models
   - Configurable timeouts and retries

2. **HuggingFace**
   - Text embeddings with various models
   - Model metadata retrieval
   - Configurable timeouts and retries

3. **Cohere**
   - Text embeddings with various models
   - Text generation for metadata
   - Configurable timeouts and retries

### Error Handling

The library provides specific exceptions for different error cases:

```python
from ai_prishtina_milvus_client.exceptions import (
    MilvusClientError,
    ConfigurationError,
    ConnectionError,
    CollectionError,
    InsertError,
    SearchError
)

try:
    with MilvusClient("config.yaml") as client:
        client.create_collection()
except ConfigurationError as e:
    print(f"Configuration error: {e}")
except ConnectionError as e:
    print(f"Connection error: {e}")
```

## Best Practices

1. **Connection Management**
   - Use context managers (`with` statement) for automatic cleanup
   - Always close connections when done
   - Handle connection errors appropriately

2. **Configuration**
   - Keep sensitive information (passwords, API keys) in environment variables
   - Use different configurations for development and production
   - Validate configuration before use

3. **Data Sources**
   - Use appropriate data source format for your use case
   - Include metadata fields for better search capabilities
   - Use batch processing for large datasets
   - Validate data before insertion

4. **Cloud Storage**
   - Use IAM roles or service accounts when possible
   - Implement proper error handling and retries
   - Use appropriate storage classes for your data
   - Monitor storage usage and costs

5. **API Integration**
   - Cache API responses when possible
   - Implement rate limiting and backoff
   - Monitor API usage and costs
   - Handle API errors gracefully

6. **Vector Operations**
   - Batch insert operations for better performance
   - Use appropriate index types for your use case
   - Monitor collection statistics
   - Clean up unused collections

7. **Error Handling**
   - Always handle exceptions appropriately
   - Log errors for debugging
   - Implement retry mechanisms for transient failures
   - Use specific exception types for better error handling

## Examples

Check the `examples/` directory for complete working examples:

- `basic_usage.py`: Simple vector operations
- `advanced_usage.py`: Working with metadata and complex queries
- `collection_management.py`: Collection management features
- `data_sources.py`: Working with different data sources
- `cloud_and_api_integration.py`: Working with cloud storage and APIs
- `audio_processing.py`: Audio feature extraction and similarity search
- `video_analysis.py`: Video frame analysis and content-based retrieval

### Audio Processing Example

```python
from ai_prishtina_milvus_client import MilvusClient
import librosa
import numpy as np

# Initialize client
with MilvusClient("config.yaml") as client:
    # Load and process audio file
    audio_path = "sample.wav"
    y, sr = librosa.load(audio_path)
    
    # Extract audio features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    
    # Combine features into a single vector
    audio_features = np.concatenate([
        mfccs.mean(axis=1),
        spectral_centroids.mean(axis=1)
    ])
    
    # Insert audio features with metadata
    metadata = {
        "filename": audio_path,
        "duration": librosa.get_duration(y=y, sr=sr),
        "sample_rate": sr
    }
    client.insert([audio_features], [metadata])
    
    # Search for similar audio files
    results = client.search(
        query_vectors=[audio_features],
        top_k=5,
        search_params={"metric_type": "L2"}
    )
```

### Video Analysis Example

```python
from ai_prishtina_milvus_client import MilvusClient
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Initialize client
with MilvusClient("config.yaml") as client:
    # Load pre-trained model for feature extraction
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    # Process video file
    video_path = "sample.mp4"
    cap = cv2.VideoCapture(video_path)
    
    frame_features = []
    frame_metadata = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Extract features from frame
        frame = cv2.resize(frame, (224, 224))
        frame = preprocess_input(frame)
        features = model.predict(np.expand_dims(frame, axis=0))[0]
        
        frame_features.append(features)
        frame_metadata.append({
            "frame_number": frame_count,
            "timestamp": frame_count / cap.get(cv2.CAP_PROP_FPS)
        })
        
        frame_count += 1
    
    cap.release()
    
    # Insert frame features with metadata
    client.insert(frame_features, frame_metadata)
    
    # Search for similar frames
    query_frame = frame_features[0]  # Example: search using first frame
    results = client.search(
        query_vectors=[query_frame],
        top_k=10,
        search_params={"metric_type": "L2"}
    )
```

### Multi-modal Analysis Example

```python
from ai_prishtina_milvus_client import MilvusClient
import librosa
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50

# Initialize client
with MilvusClient("config.yaml") as client:
    # Process video with audio
    video_path = "sample.mp4"
    
    # Extract audio features
    audio, sr = librosa.load(video_path)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    audio_features = mfccs.mean(axis=1)
    
    # Extract video features
    cap = cv2.VideoCapture(video_path)
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    frame_features = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        features = model.predict(np.expand_dims(frame, axis=0))[0]
        frame_features.append(features)
    
    cap.release()
    
    # Combine audio and video features
    combined_features = np.concatenate([
        audio_features,
        np.mean(frame_features, axis=0)
    ])
    
    # Insert with metadata
    metadata = {
        "filename": video_path,
        "duration": librosa.get_duration(y=audio, sr=sr),
        "frame_count": len(frame_features)
    }
    client.insert([combined_features], [metadata])
    
    # Search for similar content
    results = client.search(
        query_vectors=[combined_features],
        top_k=5,
        search_params={"metric_type": "L2"}
    )
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details. 