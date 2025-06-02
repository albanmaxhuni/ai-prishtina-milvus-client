"""
Unit tests for data management module.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from ai_prishtina_milvus_client import (
    DataManager,
    DataValidationConfig,
    DataCleaningConfig,
    DataTransformationConfig
)

def test_validate_data(data_validation_config):
    """Test data validation."""
    manager = DataManager(validation_config=data_validation_config)
    
    # Test successful validation
    data = [
        {"id": 1, "text": "test", "category": "test", "score": 0.9},
        {"id": 2, "text": "test2", "category": "test", "score": 0.8}
    ]
    valid_data, errors = manager.validate_data(data)
    assert len(valid_data) == 2
    assert len(errors) == 0
    
    # Test validation with errors
    invalid_data = [
        {"id": 1},  # Missing required fields
        {"id": 2, "text": 123, "category": "test", "score": "invalid"}  # Wrong types
    ]
    valid_data, errors = manager.validate_data(invalid_data)
    assert len(valid_data) == 0
    assert len(errors) > 0

def test_clean_data(data_cleaning_config):
    """Test data cleaning."""
    manager = DataManager(cleaning_config=data_cleaning_config)
    
    # Test successful cleaning
    data = [
        {"id": 1, "text": "test", "category": "test", "score": 0.9},
        {"id": 1, "text": "test", "category": "test", "score": 0.9},  # Duplicate
        {"id": 2, "text": "test2", "category": "test", "score": None}  # Missing value
    ]
    cleaned_data = manager.clean_data(data)
    assert len(cleaned_data) == 2  # Duplicate removed
    assert all("score" in record for record in cleaned_data)
    assert all(record["score"] is not None for record in cleaned_data)
    
    # Test cleaning with empty data
    cleaned_data = manager.clean_data([])
    assert len(cleaned_data) == 0

def test_transform_data(data_transformation_config):
    """Test data transformation."""
    manager = DataManager(transformation_config=data_transformation_config)
    
    # Test successful transformation
    data = [
        {
            "id": 1,
            "text": "test",
            "category": "test",
            "score": 0.9,
            "vector": np.random.rand(128)
        }
    ]
    transformed_data = manager.transform_data(data)
    assert len(transformed_data) == 1
    assert "normalized_vector" in transformed_data[0]
    assert "metadata" in transformed_data[0]
    
    # Test transformation with empty data
    transformed_data = manager.transform_data([])
    assert len(transformed_data) == 0

def test_export_data(tmp_path):
    """Test data export."""
    manager = DataManager()
    
    # Test successful export
    data = [
        {"id": 1, "text": "test", "category": "test", "score": 0.9},
        {"id": 2, "text": "test2", "category": "test", "score": 0.8}
    ]
    file_path = tmp_path / "test_data.json"
    manager.export_data(data, str(file_path))
    assert file_path.exists()
    
    # Test export with empty data
    file_path = tmp_path / "empty_data.json"
    manager.export_data([], str(file_path))
    assert file_path.exists()

def test_import_data(tmp_path):
    """Test data import."""
    manager = DataManager()
    
    # Test successful import
    data = [
        {"id": 1, "text": "test", "category": "test", "score": 0.9},
        {"id": 2, "text": "test2", "category": "test", "score": 0.8}
    ]
    file_path = tmp_path / "test_data.json"
    manager.export_data(data, str(file_path))
    imported_data = manager.import_data(str(file_path))
    assert len(imported_data) == 2
    assert imported_data == data
    
    # Test import with non-existent file
    with pytest.raises(FileNotFoundError):
        manager.import_data("non_existent.json")

def test_data_validation_config():
    """Test data validation configuration."""
    config = DataValidationConfig(
        required_fields=["id", "text", "category"],
        field_types={
            "id": "int",
            "text": "str",
            "category": "str",
            "score": "float"
        },
        value_ranges={
            "score": (0.0, 1.0)
        }
    )
    
    assert len(config.required_fields) == 3
    assert len(config.field_types) == 4
    assert "score" in config.value_ranges

def test_data_cleaning_config():
    """Test data cleaning configuration."""
    config = DataCleaningConfig(
        remove_duplicates=True,
        fill_missing_values=True,
        normalize_numeric_fields=True,
        remove_outliers=True
    )
    
    assert config.remove_duplicates is True
    assert config.fill_missing_values is True
    assert config.normalize_numeric_fields is True
    assert config.remove_outliers is True

def test_data_transformation_config():
    """Test data transformation configuration."""
    config = DataTransformationConfig(
        field_mappings={
            "old_field": "new_field"
        },
        vector_normalization=True,
        metadata_extraction={"field": "rule"}
    )
    
    assert len(config.field_mappings) == 1
    assert config.vector_normalization is True
    assert config.metadata_extraction == {"field": "rule"} 