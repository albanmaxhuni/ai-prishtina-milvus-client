"""
Custom exceptions for the Milvus client.
"""

class MilvusClientError(Exception):
    """Base exception for all Milvus client errors."""
    pass


class ConfigurationError(MilvusClientError):
    """Raised when there is an error in the configuration."""
    pass


class ConnectionError(MilvusClientError):
    """Raised when there is an error connecting to Milvus."""
    pass


class CollectionError(MilvusClientError):
    """Raised when there is an error with collection operations."""
    pass


class InsertError(MilvusClientError):
    """Raised when there is an error inserting data."""
    pass


class SearchError(MilvusClientError):
    """Raised when there is an error searching data."""
    pass


class IndexError(MilvusClientError):
    """Raised when there is an error with index operations."""
    pass 