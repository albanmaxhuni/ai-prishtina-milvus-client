"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from ai_prishtina_milvus_client import (
    MilvusClient,
    DataValidator,
    VectorValidationConfig,
    BatchConfig,
    MonitoringConfig,
    RetryConfig,
    BackupConfig,
    SecurityConfig,
    SearchConfig,
    DataValidationConfig,
    DataCleaningConfig,
    DataTransformationConfig,
    PerformanceConfig,
    LoggingConfig,
    DebugConfig,
    TestConfig
)

@pytest.fixture
def sample_vectors() -> List[List[float]]:
    """Generate sample vectors for testing."""
    return np.random.rand(10, 128).tolist()

@pytest.fixture
def sample_metadata() -> List[Dict[str, Any]]:
    """Generate sample metadata for testing."""
    return [
        {
            "id": i,
            "text": f"Sample text {i}",
            "category": f"Category {i % 3}",
            "score": float(i) / 10.0
        }
        for i in range(10)
    ]

@pytest.fixture
def vector_validation_config() -> VectorValidationConfig:
    """Create vector validation configuration."""
    return VectorValidationConfig(
        expected_dim=128,
        normalize=True,
        check_type=True
    )

@pytest.fixture
def batch_config() -> BatchConfig:
    """Create batch configuration."""
    return BatchConfig(
        batch_size=1000,
        max_workers=4,
        show_progress=True
    )

@pytest.fixture
def monitoring_config() -> MonitoringConfig:
    """Create monitoring configuration."""
    return MonitoringConfig(
        collect_system_metrics=True,
        metrics_history_size=1000,
        collection_interval=1.0
    )

@pytest.fixture
def retry_config() -> RetryConfig:
    """Create retry configuration."""
    return RetryConfig(
        max_retries=3,
        initial_delay=1.0,
        max_delay=60.0
    )

@pytest.fixture
def backup_config() -> BackupConfig:
    """Create backup configuration."""
    return BackupConfig(
        backup_dir="backups",
        max_backups=5,
        backup_interval=3600.0
    )

@pytest.fixture
def security_config() -> SecurityConfig:
    """Create security configuration."""
    return SecurityConfig(
        secret_key="test_secret_key",
        token_expiry=3600,
        require_ssl=True
    )

@pytest.fixture
def search_config() -> SearchConfig:
    """Create search configuration."""
    return SearchConfig(
        metric_type="L2",
        top_k=10,
        nprobe=10
    )

@pytest.fixture
def data_validation_config() -> DataValidationConfig:
    """Create data validation configuration."""
    return DataValidationConfig(
        required_fields=["id", "text"],
        field_types={"id": "int", "score": "float"}
    )

@pytest.fixture
def data_cleaning_config() -> DataCleaningConfig:
    """Create data cleaning configuration."""
    return DataCleaningConfig(
        remove_duplicates=True,
        fill_missing=True,
        normalize=True
    )

@pytest.fixture
def data_transformation_config() -> DataTransformationConfig:
    """Create data transformation configuration."""
    return DataTransformationConfig(
        vector_normalization=True,
        field_mappings={"old_field": "new_field"}
    )

@pytest.fixture
def performance_config() -> PerformanceConfig:
    """Create performance configuration."""
    return PerformanceConfig(
        use_threading=True,
        use_multiprocessing=False
    )

@pytest.fixture
def logging_config() -> LoggingConfig:
    """Create logging configuration."""
    return LoggingConfig(
        level="INFO",
        file="test.log"
    )

@pytest.fixture
def debug_config() -> DebugConfig:
    """Create debug configuration."""
    return DebugConfig(
        enabled=True,
        break_on_error=False,
        trace_calls=True
    )

@pytest.fixture
def test_config() -> TestConfig:
    """Create test configuration."""
    return TestConfig(
        test_dir="tests",
        pattern="test_*.py",
        coverage=True
    ) 