"""
AI Prishtina Milvus Client
=========================

A high-level Python library for interacting with Milvus vector database.
Provides a simplified interface for common operations and handles connection
management, error handling, and configuration.

Author: Alban Maxhuni, PhD
Email: alban.q.maxhuni@gmail.com | info@albanmaxhuni.com
License: MIT
"""

from ai_prishtina_milvus_client.client import MilvusClient
from ai_prishtina_milvus_client.config import MilvusConfig
from ai_prishtina_milvus_client.exceptions import (
    MilvusClientError,
    ConfigurationError,
    ConnectionError,
    CollectionError,
    InsertError,
    SearchError
)
from ai_prishtina_milvus_client.data_sources import (
    DataSource,
    CSVDataSource,
    JSONDataSource,
    NumPyDataSource
)
from ai_prishtina_milvus_client.cloud_storage import (
    CloudStorage,
    S3Storage,
    GCSStorage,
    AzureStorage
)
from ai_prishtina_milvus_client.api_integrations import (
    APIClient,
    OpenAIClient,
    HuggingFaceClient,
    CohereClient
)
from ai_prishtina_milvus_client.streaming import StreamingClient
from ai_prishtina_milvus_client.distributed import DistributedClient
from ai_prishtina_milvus_client.advanced import AdvancedClient

__version__ = "0.1.0"
__author__ = "Alban Maxhuni, PhD"
__email__ = "alban.q.maxhuni@gmail.com"
__license__ = "MIT"

__all__ = [
    # Core components
    "MilvusClient",
    "MilvusConfig",
    
    # Exceptions
    "MilvusClientError",
    "ConfigurationError",
    "ConnectionError",
    "CollectionError",
    "InsertError",
    "SearchError",
    
    # Data sources
    "DataSource",
    "CSVDataSource",
    "JSONDataSource",
    "NumPyDataSource",
    
    # Cloud storage
    "CloudStorage",
    "S3Storage",
    "GCSStorage",
    "AzureStorage",
    
    # API integrations
    "APIClient",
    "OpenAIClient",
    "HuggingFaceClient",
    "CohereClient",
    
    # Advanced features
    "StreamingClient",
    "DistributedClient",
    "AdvancedClient"
] 