"""
Data source implementations for various file formats.
"""

import json
import pickle
import csv
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import h5py
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field


class DataSourceConfig(BaseModel):
    """Configuration for data sources."""
    type: str = Field(..., description="Data source type")
    path: str = Field(..., description="Path to the data file")
    vector_field: str = Field(..., description="Name of the vector field")
    metadata_fields: Optional[List[str]] = Field(None, description="Names of metadata fields")
    batch_size: int = Field(1000, description="Number of items to process at once")
    format_specific: Optional[Dict[str, Any]] = Field(None, description="Format-specific parameters")


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        
    @abstractmethod
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load vectors and metadata from the data source."""
        pass


class CSVDataSource(DataSource):
    """CSV file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from CSV file."""
        try:
            df = pd.read_csv(self.config.path)
            vectors = [eval(v) if isinstance(v, str) else v for v in df[self.config.vector_field]]
            
            metadata = None
            if self.config.metadata_fields:
                metadata = df[self.config.metadata_fields].to_dict('records')
                
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load CSV data: {str(e)}")


class JSONDataSource(DataSource):
    """JSON file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from JSON file."""
        try:
            with open(self.config.path) as f:
                data = json.load(f)
                
            if isinstance(data, list):
                vectors = [item[self.config.vector_field] for item in data]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: item[k] for k in self.config.metadata_fields} for item in data]
            else:
                vectors = data[self.config.vector_field]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: data[k] for k in self.config.metadata_fields}]
                    
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load JSON data: {str(e)}")


class NumPyDataSource(DataSource):
    """NumPy file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from NumPy file."""
        try:
            data = np.load(self.config.path)
            vectors = data[self.config.vector_field].tolist()
            
            metadata = None
            if self.config.metadata_fields:
                metadata = [{k: data[k].tolist() for k in self.config.metadata_fields}]
                
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load NumPy data: {str(e)}")


class HDF5DataSource(DataSource):
    """HDF5 file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from HDF5 file."""
        try:
            with h5py.File(self.config.path, 'r') as f:
                vectors = f[self.config.vector_field][:].tolist()
                
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: f[k][:].tolist() for k in self.config.metadata_fields}]
                    
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load HDF5 data: {str(e)}")


class ParquetDataSource(DataSource):
    """Parquet file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from Parquet file."""
        try:
            df = pd.read_parquet(self.config.path)
            vectors = df[self.config.vector_field].tolist()
            
            metadata = None
            if self.config.metadata_fields:
                metadata = df[self.config.metadata_fields].to_dict('records')
                
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load Parquet data: {str(e)}")


class PickleDataSource(DataSource):
    """Pickle file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from Pickle file."""
        try:
            with open(self.config.path, 'rb') as f:
                data = pickle.load(f)
                
            if isinstance(data, dict):
                vectors = data[self.config.vector_field]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: data[k] for k in self.config.metadata_fields}]
            else:
                vectors = [item[self.config.vector_field] for item in data]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: item[k] for k in self.config.metadata_fields} for item in data]
                    
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load Pickle data: {str(e)}")


class YAMLDataSource(DataSource):
    """YAML file data source."""
    
    def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from YAML file."""
        try:
            with open(self.config.path) as f:
                data = yaml.safe_load(f)
                
            if isinstance(data, list):
                vectors = [item[self.config.vector_field] for item in data]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: item[k] for k in self.config.metadata_fields} for item in data]
            else:
                vectors = data[self.config.vector_field]
                metadata = None
                if self.config.metadata_fields:
                    metadata = [{k: data[k] for k in self.config.metadata_fields}]
                    
            return vectors, metadata
        except Exception as e:
            raise Exception(f"Failed to load YAML data: {str(e)}")


class DataSourceFactory:
    """Factory for creating data sources."""
    
    _sources = {
        "csv": CSVDataSource,
        "json": JSONDataSource,
        "numpy": NumPyDataSource,
        "hdf5": HDF5DataSource,
        "parquet": ParquetDataSource,
        "pickle": PickleDataSource,
        "yaml": YAMLDataSource,
    }
    
    @classmethod
    def create(cls, config: DataSourceConfig) -> DataSource:
        """
        Create a data source instance.
        
        Args:
            config: Data source configuration
            
        Returns:
            DataSource instance
            
        Raises:
            ValueError: If source type is not supported
        """
        source_class = cls._sources.get(config.type.lower())
        if not source_class:
            raise ValueError(f"Unsupported data source type: {config.type}")
        return source_class(config)


def load_data_source(config_path: str) -> DataSource:
    """
    Load data source from configuration file.
    
    Args:
        config_path: Path to the data source configuration file
        
    Returns:
        DataSource instance
    """
    config_path = Path(config_path)
    
    # Load configuration from file
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config_data = json.load(f)
            
    # Create and validate configuration
    config = DataSourceConfig(**config_data)
    return DataSourceFactory.create(config) 