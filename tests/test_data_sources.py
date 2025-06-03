"""
Unit tests for data sources.
"""

import json
import os
import pickle
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from ai_prishtina_milvus_client.data_sources import (
    DataSourceConfig,
    DataSourceFactory,
    load_data_source,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    vectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    metadata = [
        {"category": "A", "score": 0.8},
        {"category": "B", "score": 0.9}
    ]
    return vectors, metadata


@pytest.fixture
def config():
    """Create sample configuration."""
    return {
        "type": "csv",  # Will be overridden in tests
        "path": "dummy_path",  # Will be overridden in tests
        "vector_field": "vector",
        "metadata_fields": ["category", "score"],
        "batch_size": 1000
    }


def test_csv_data_source(sample_data, config):
    """Test CSV data source."""
    vectors, metadata = sample_data
    
    # Create CSV file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            "vector": [str(v) for v in vectors],
            "category": [m["category"] for m in metadata],
            "score": [m["score"] for m in metadata]
        })
        df.to_csv(f.name, index=False)
        
        # Update config
        config["type"] = "csv"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_json_data_source(sample_data, config):
    """Test JSON data source."""
    vectors, metadata = sample_data
    
    # Create JSON file
    with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as f:
        data = [
            {"vector": v, **m}
            for v, m in zip(vectors, metadata)
        ]
        json.dump(data, f)
        
        # Update config
        config["type"] = "json"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_numpy_data_source(sample_data, config):
    """Test NumPy data source."""
    vectors, metadata = sample_data
    
    # Create NumPy file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(
            f.name,
            vector=np.array(vectors),
            category=np.array([m["category"] for m in metadata]),
            score=np.array([m["score"] for m in metadata])
        )
        
        # Update config
        config["type"] = "numpy"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_hdf5_data_source(sample_data, config):
    """Test HDF5 data source."""
    vectors, metadata = sample_data
    
    # Create HDF5 file
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        with h5py.File(f.name, 'w') as hf:
            hf.create_dataset('vector', data=vectors)
            hf.create_dataset('category', data=[m["category"] for m in metadata])
            hf.create_dataset('score', data=[m["score"] for m in metadata])
            
        # Update config
        config["type"] = "hdf5"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_parquet_data_source(sample_data, config):
    """Test Parquet data source."""
    vectors, metadata = sample_data
    
    # Create Parquet file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df = pd.DataFrame({
            "vector": vectors,
            "category": [m["category"] for m in metadata],
            "score": [m["score"] for m in metadata]
        })
        df.to_parquet(f.name)
        
        # Update config
        config["type"] = "parquet"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_pickle_data_source(sample_data, config):
    """Test Pickle data source."""
    vectors, metadata = sample_data
    
    # Create Pickle file
    with tempfile.NamedTemporaryFile(suffix='.pkl', mode='wb', delete=False) as f:
        data = [
            {"vector": v, **m}
            for v, m in zip(vectors, metadata)
        ]
        pickle.dump(data, f)
        
        # Update config
        config["type"] = "pickle"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_yaml_data_source(sample_data, config):
    """Test YAML data source."""
    vectors, metadata = sample_data
    
    # Create YAML file
    with tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as f:
        data = [
            {"vector": v, **m}
            for v, m in zip(vectors, metadata)
        ]
        yaml.dump(data, f)
        
        # Update config
        config["type"] = "yaml"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(f.name)


def test_load_data_source(sample_data, config):
    """Test loading data source from configuration file."""
    vectors, metadata = sample_data
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as data_file, \
         tempfile.NamedTemporaryFile(suffix='.yaml', mode='w', delete=False) as config_file:
        
        # Create data file
        df = pd.DataFrame({
            "vector": [str(v) for v in vectors],
            "category": [m["category"] for m in metadata],
            "score": [m["score"] for m in metadata]
        })
        df.to_csv(data_file.name, index=False)
        
        # Create config file
        config["type"] = "csv"
        config["path"] = data_file.name
        yaml.dump(config, config_file)
        
        # Test loading
        source = load_data_source(config_file.name)
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata == metadata
        
    os.unlink(data_file.name)
    os.unlink(config_file.name)


def test_invalid_data_source_type(config):
    """Test invalid data source type."""
    config["type"] = "invalid"
    
    with pytest.raises(ValueError):
        DataSourceFactory.create(DataSourceConfig(**config))


def test_missing_vector_field(sample_data, config):
    """Test missing vector field."""
    vectors, metadata = sample_data
    
    # Create CSV file without vector field
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            "category": [m["category"] for m in metadata],
            "score": [m["score"] for m in metadata]
        })
        df.to_csv(f.name, index=False)
        
        # Update config
        config["type"] = "csv"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        with pytest.raises(Exception):
            source.load_data()
            
    os.unlink(f.name)


def test_missing_metadata_fields(sample_data, config):
    """Test missing metadata fields."""
    vectors, metadata = sample_data
    
    # Create CSV file without metadata fields
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        df = pd.DataFrame({
            "vector": [str(v) for v in vectors]
        })
        df.to_csv(f.name, index=False)
        
        # Update config
        config["type"] = "csv"
        config["path"] = f.name
        
        # Test loading
        source = DataSourceFactory.create(DataSourceConfig(**config))
        loaded_vectors, loaded_metadata = source.load_data()
        
        assert loaded_vectors == vectors
        assert loaded_metadata is None
        
    os.unlink(f.name) 