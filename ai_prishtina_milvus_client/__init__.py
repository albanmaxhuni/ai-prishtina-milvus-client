"""
AI Prishtina Milvus Client - A comprehensive client for Milvus vector database operations.
"""

from .client import MilvusClient
from .config import MilvusConfig
from .data_validation import DataValidator, VectorValidationConfig
from .batch_operations import BatchProcessor, BatchConfig, BatchMetrics
from .monitoring import MetricsCollector, MonitoringConfig, PerformanceMetrics, SystemMetrics
from .error_recovery import ErrorRecovery, RetryConfig, BackupConfig
from .security import SecurityManager, SecurityConfig, User
from .advanced_search import AdvancedSearch, SearchConfig, FilterConfig, RankingConfig, FilterCondition
from .data_management import DataManager, DataValidationConfig, DataCleaningConfig, DataTransformationConfig
from .performance import PerformanceOptimizer, PerformanceConfig, CacheConfig, BatchConfig
from .dev_tools import DevTools, LoggingConfig, DebugConfig, TestConfig
from .advanced import AdvancedMilvusClient, PartitionConfig, HybridQueryConfig
from .streaming import StreamConfig, StreamMessage, KafkaStreamProcessor

__version__ = "0.1.0"

__all__ = [
    # Core client
    "MilvusClient",
    "MilvusConfig",
    "AdvancedMilvusClient",
    "PartitionConfig",
    "HybridQueryConfig",
    
    # Streaming
    "StreamConfig",
    "StreamMessage",
    "KafkaStreamProcessor",
    
    # Data validation
    "DataValidator",
    "VectorValidationConfig",
    
    # Batch operations
    "BatchProcessor",
    "BatchConfig",
    "BatchMetrics",
    
    # Monitoring
    "MetricsCollector",
    "MonitoringConfig",
    "PerformanceMetrics",
    "SystemMetrics",
    
    # Error recovery
    "ErrorRecovery",
    "RetryConfig",
    "BackupConfig",
    
    # Security
    "SecurityManager",
    "SecurityConfig",
    "User",
    
    # Advanced search
    "AdvancedSearch",
    "SearchConfig",
    "FilterConfig",
    "RankingConfig",
    "FilterCondition",
    
    # Data management
    "DataManager",
    "DataValidationConfig",
    "DataCleaningConfig",
    "DataTransformationConfig",
    
    # Performance
    "PerformanceOptimizer",
    "PerformanceConfig",
    "CacheConfig",
    "BatchConfig",
    
    # Development tools
    "DevTools",
    "LoggingConfig",
    "DebugConfig",
    "TestConfig",
] 