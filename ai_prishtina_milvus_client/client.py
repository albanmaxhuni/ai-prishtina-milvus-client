"""
Main Milvus client implementation with both synchronous and asynchronous interfaces.
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
from functools import wraps

import numpy as np
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from ai_prishtina_milvus_client.config import MilvusConfig
from ai_prishtina_milvus_client.data_sources import DataSource, load_data_source
from ai_prishtina_milvus_client.cloud_storage import CloudStorage, load_cloud_storage
from ai_prishtina_milvus_client.api_integrations import APIClient, load_api_client
from ai_prishtina_milvus_client.exceptions import (
    CollectionError,
    ConnectionError,
    InsertError,
    SearchError,
)

def async_operation(func):
    """Decorator to run synchronous operations in a thread pool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper

class AsyncMilvusClient:
    """
    An asynchronous high-level client for interacting with Milvus.
    
    This client provides an async interface for common Milvus operations
    and handles connection management and error handling.
    """
    
    def __init__(self, config: Union[str, MilvusConfig]):
        """
        Initialize the async Milvus client.
        
        Args:
            config: Either a path to the YAML configuration file or a MilvusConfig instance
            
        Raises:
            ConfigurationError: If the configuration is invalid
            ConnectionError: If connection to Milvus fails
        """
        if isinstance(config, str):
            self.config = MilvusConfig.from_yaml(config)
        else:
            self.config = config
        self._collection = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    @async_operation
    async def _connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db_name=self.config.db_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
            
    async def connect(self) -> None:
        """Async wrapper for connection."""
        await self._connect()
            
    @async_operation
    async def _get_collection(self) -> Collection:
        """Get or create the collection."""
        if self._collection is None:
            try:
                if utility.has_collection(self.config.collection_name):
                    self._collection = Collection(self.config.collection_name)
                else:
                    await self.create_collection()
            except Exception as e:
                raise CollectionError(f"Failed to get collection: {str(e)}")
        return self._collection
        
    @async_operation
    async def create_collection(self) -> None:
        """
        Create a new collection with the configured schema.
        
        Raises:
            CollectionError: If collection creation fails
        """
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dim),
            ]
            # Add metadata fields if present
            if getattr(self.config, 'metadata_fields', None):
                type_map = {
                    "int": DataType.INT64,
                    "int64": DataType.INT64,
                    "float": DataType.FLOAT,
                    "float32": DataType.FLOAT,
                    "double": DataType.DOUBLE,
                    "float64": DataType.DOUBLE,
                    "str": DataType.VARCHAR,
                    "string": DataType.VARCHAR,
                    "bool": DataType.BOOL,
                    "json": DataType.JSON,
                }
                for field in self.config.metadata_fields:
                    field_name = field["name"]
                    field_type = field["type"].lower()
                    dtype = type_map.get(field_type, DataType.VARCHAR)
                    if dtype == DataType.VARCHAR:
                        fields.append(FieldSchema(name=field_name, dtype=dtype, max_length=512))
                    else:
                        fields.append(FieldSchema(name=field_name, dtype=dtype))
            schema = CollectionSchema(fields=fields, description="Vector collection")
            self._collection = Collection(
                name=self.config.collection_name,
                schema=schema,
                using="default",
            )
            
            # Create index
            index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist},
            }
            self._collection.create_index(field_name="vector", index_params=index_params)
            
        except Exception as e:
            raise CollectionError(f"Failed to create collection: {str(e)}")
            
    @async_operation
    async def insert(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Insert vectors into the collection.
        
        Args:
            vectors: List of vectors to insert
            metadata: Optional list of metadata dictionaries for each vector
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            collection = await self._get_collection()
            data = [vectors]
            if metadata:
                for key in metadata[0].keys():
                    data.append([m[key] for m in metadata])
            collection.insert(data)
            collection.flush()
        except Exception as e:
            raise InsertError(f"Failed to insert vectors: {str(e)}")
            
    async def insert_from_source(self, source: Union[str, DataSource]) -> None:
        """
        Insert vectors from a data source.
        
        Args:
            source: Path to data source config file or DataSource instance
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            if isinstance(source, str):
                source = load_data_source(source)
                
            vectors, metadata = await source.load_data()
            await self.insert(vectors, metadata)
            
        except Exception as e:
            raise InsertError(f"Failed to insert from data source: {str(e)}")
            
    async def insert_from_cloud(self, cloud_config: str, remote_path: str) -> None:
        """
        Insert vectors from cloud storage.
        
        Args:
            cloud_config: Path to cloud storage configuration file
            remote_path: Path to the file in cloud storage
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            # Load cloud storage client
            cloud = load_cloud_storage(cloud_config)
            await cloud.connect()
            
            # Download file to temporary location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                await cloud.download_file(remote_path, tmp.name)
                
                # Load data source from downloaded file
                source = load_data_source(tmp.name)
                vectors, metadata = await source.load_data()
                
                # Insert vectors
                await self.insert(vectors, metadata)
                
        except Exception as e:
            raise InsertError(f"Failed to insert from cloud storage: {str(e)}")
            
    async def insert_from_api(self, api_config: str, query: str, **kwargs) -> None:
        """
        Insert vectors from API service.
        
        Args:
            api_config: Path to API configuration file
            query: Query string for the API
            **kwargs: Additional arguments for the API client
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            # Load API client
            api = load_api_client(api_config)
            
            # Get vectors and metadata from API
            vectors = await api.get_vectors(query, **kwargs)
            metadata = await api.get_metadata(query, **kwargs)
            
            # Insert vectors
            await self.insert(vectors, metadata)
            
        except Exception as e:
            raise InsertError(f"Failed to insert from API: {str(e)}")
            
    @async_operation
    async def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of nearest neighbors to return
            search_params: Optional search parameters
            
        Returns:
            List of search results for each query vector
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = await self._get_collection()
            collection.load()
            
            search_params = search_params or {}
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id"] + [f["name"] for f in self.config.metadata_fields]
            )
            
            return [[hit.entity.to_dict() for hit in hits] for hits in results]
            
        except Exception as e:
            raise SearchError(f"Failed to search vectors: {str(e)}")
            
    @async_operation
    async def delete(self, expr: str) -> None:
        """
        Delete vectors matching the expression.
        
        Args:
            expr: Expression to match vectors for deletion
            
        Raises:
            CollectionError: If deletion fails
        """
        try:
            collection = await self._get_collection()
            collection.delete(expr)
        except Exception as e:
            raise CollectionError(f"Failed to delete vectors: {str(e)}")
            
    @async_operation
    async def close(self) -> None:
        """Close the connection to Milvus."""
        try:
            connections.disconnect("default")
        except Exception:
            pass
            
    @async_operation
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary containing collection statistics
            
        Raises:
            CollectionError: If getting stats fails
        """
        try:
            collection = await self._get_collection()
            return collection.get_statistics()
        except Exception as e:
            raise CollectionError(f"Failed to get collection statistics: {str(e)}")
            
    @async_operation
    async def drop_collection(self) -> None:
        """
        Drop the collection.
        
        Raises:
            CollectionError: If dropping collection fails
        """
        try:
            utility.drop_collection(self.config.collection_name)
            self._collection = None
        except Exception as e:
            raise CollectionError(f"Failed to drop collection: {str(e)}")
            
    @async_operation
    async def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Returns:
            List of collection names
        """
        return utility.list_collections()

# Keep the original synchronous client for backward compatibility
class MilvusClient:
    """Synchronous Milvus client implementation."""
    
    def __init__(self, config: Union[str, MilvusConfig]):
        """
        Initialize the Milvus client.
        
        Args:
            config: Either a path to the YAML configuration file or a MilvusConfig instance
            
        Raises:
            ConfigurationError: If the configuration is invalid
            ConnectionError: If connection to Milvus fails
        """
        if isinstance(config, str):
            self.config = MilvusConfig.from_yaml(config)
        else:
            self.config = config
        self._connect()
        self._collection = None
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        
    def _connect(self) -> None:
        """Establish connection to Milvus server."""
        try:
            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db_name=self.config.db_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {str(e)}")
            
    def _get_collection(self) -> Collection:
        """Get or create the collection."""
        if self._collection is None:
            try:
                if utility.has_collection(self.config.collection_name):
                    self._collection = Collection(self.config.collection_name)
                else:
                    self.create_collection()
            except Exception as e:
                raise CollectionError(f"Failed to get collection: {str(e)}")
        return self._collection
        
    def create_collection(self) -> None:
        """
        Create a new collection with the configured schema.
        
        Raises:
            CollectionError: If collection creation fails
        """
        try:
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.config.dim),
            ]
            # Add metadata fields if present
            if getattr(self.config, 'metadata_fields', None):
                type_map = {
                    "int": DataType.INT64,
                    "int64": DataType.INT64,
                    "float": DataType.FLOAT,
                    "float32": DataType.FLOAT,
                    "double": DataType.DOUBLE,
                    "float64": DataType.DOUBLE,
                    "str": DataType.VARCHAR,
                    "string": DataType.VARCHAR,
                    "bool": DataType.BOOL,
                    "json": DataType.JSON,
                }
                for field in self.config.metadata_fields:
                    field_name = field["name"]
                    field_type = field["type"].lower()
                    dtype = type_map.get(field_type, DataType.VARCHAR)
                    if dtype == DataType.VARCHAR:
                        fields.append(FieldSchema(name=field_name, dtype=dtype, max_length=512))
                    else:
                        fields.append(FieldSchema(name=field_name, dtype=dtype))
            schema = CollectionSchema(fields=fields, description="Vector collection")
            self._collection = Collection(
                name=self.config.collection_name,
                schema=schema,
                using="default",
            )
            
            # Create index
            index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": {"nlist": self.config.nlist},
            }
            self._collection.create_index(field_name="vector", index_params=index_params)
            
        except Exception as e:
            raise CollectionError(f"Failed to create collection: {str(e)}")
            
    def insert(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Insert vectors into the collection.
        
        Args:
            vectors: List of vectors to insert
            metadata: Optional list of metadata dictionaries for each vector
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            collection = self._get_collection()
            data = [vectors]
            if metadata:
                for key in metadata[0].keys():
                    data.append([m[key] for m in metadata])
            collection.insert(data)
            collection.flush()
        except Exception as e:
            raise InsertError(f"Failed to insert vectors: {str(e)}")
            
    def insert_from_source(self, source: Union[str, DataSource]) -> None:
        """
        Insert vectors from a data source.
        
        Args:
            source: Path to data source config file or DataSource instance
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            if isinstance(source, str):
                source = load_data_source(source)
                
            vectors, metadata = source.load_data()
            self.insert(vectors, metadata)
            
        except Exception as e:
            raise InsertError(f"Failed to insert from data source: {str(e)}")
            
    def insert_from_cloud(self, cloud_config: str, remote_path: str) -> None:
        """
        Insert vectors from cloud storage.
        
        Args:
            cloud_config: Path to cloud storage configuration file
            remote_path: Path to the file in cloud storage
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            # Load cloud storage client
            cloud = load_cloud_storage(cloud_config)
            cloud.connect()
            
            # Download file to temporary location
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                cloud.download_file(remote_path, tmp.name)
                
                # Load data source from downloaded file
                source = load_data_source(tmp.name)
                vectors, metadata = source.load_data()
                
                # Insert vectors
                self.insert(vectors, metadata)
                
        except Exception as e:
            raise InsertError(f"Failed to insert from cloud storage: {str(e)}")
            
    def insert_from_api(self, api_config: str, query: str, **kwargs) -> None:
        """
        Insert vectors from API service.
        
        Args:
            api_config: Path to API configuration file
            query: Query string for the API
            **kwargs: Additional arguments for the API client
            
        Raises:
            InsertError: If insertion fails
        """
        try:
            # Load API client
            api = load_api_client(api_config)
            
            # Get vectors and metadata from API
            vectors = api.get_vectors(query, **kwargs)
            metadata = api.get_metadata(query, **kwargs)
            
            # Insert vectors
            self.insert(vectors, metadata)
            
        except Exception as e:
            raise InsertError(f"Failed to insert from API: {str(e)}")
            
    def search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for similar vectors.
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of results to return per query
            search_params: Optional search parameters
            
        Returns:
            List of results for each query vector
            
        Raises:
            SearchError: If search fails
        """
        try:
            collection = self._get_collection()
            collection.load()
            
            if search_params is None:
                search_params = {"metric_type": self.config.metric_type, "params": {"nprobe": 10}}
                
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["id"],
            )
            
            return [
                [{"id": hit.id, "distance": hit.distance} for hit in result]
                for result in results
            ]
            
        except Exception as e:
            raise SearchError(f"Failed to search vectors: {str(e)}")
            
    def delete(self, expr: str) -> None:
        """
        Delete vectors from the collection.
        
        Args:
            expr: Expression to filter vectors to delete
            
        Raises:
            CollectionError: If deletion fails
        """
        try:
            collection = self._get_collection()
            collection.delete(expr)
        except Exception as e:
            raise CollectionError(f"Failed to delete vectors: {str(e)}")
            
    def close(self) -> None:
        """Close the connection to Milvus."""
        try:
            connections.disconnect("default")
        except Exception:
            pass
            
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary containing collection statistics
            
        Raises:
            CollectionError: If getting statistics fails
        """
        try:
            collection = self._get_collection()
            return collection.get_statistics()
        except Exception as e:
            raise CollectionError(f"Failed to get collection statistics: {str(e)}")
            
    def drop_collection(self) -> None:
        """
        Drop the collection.
        
        Raises:
            CollectionError: If dropping the collection fails
        """
        try:
            if utility.has_collection(self.config.collection_name):
                utility.drop_collection(self.config.collection_name)
                self._collection = None
        except Exception as e:
            raise CollectionError(f"Failed to drop collection: {str(e)}")
            
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
            
        Raises:
            CollectionError: If listing collections fails
        """
        try:
            return utility.list_collections()
        except Exception as e:
            raise CollectionError(f"Failed to list collections: {str(e)}") 