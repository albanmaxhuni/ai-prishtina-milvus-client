"""
Data source implementations for various file formats with async support.
"""

import json
import pickle
import csv
import aiofiles
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import h5py
import pyarrow.parquet as pq
import yaml
from pydantic import BaseModel, Field

from ai_prishtina_milvus_client.exceptions import DataSourceError


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
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load vectors and metadata from the data source asynchronously."""
        pass


class CSVDataSource(DataSource):
    """CSV file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from CSV file asynchronously."""
        try:
            # Use asyncio.to_thread for pandas operations
            df = await asyncio.to_thread(pd.read_csv, self.config.path)
            vectors = [eval(v) if isinstance(v, str) else v for v in df[self.config.vector_field]]
            
            metadata = None
            if self.config.metadata_fields:
                metadata = df[self.config.metadata_fields].to_dict('records')
                
            return vectors, metadata
        except Exception as e:
            raise DataSourceError(f"Failed to load CSV data: {str(e)}")


class JSONDataSource(DataSource):
    """JSON file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from JSON file asynchronously."""
        try:
            async with aiofiles.open(self.config.path, mode='r') as f:
                content = await f.read()
                data = json.loads(content)
                
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
            raise DataSourceError(f"Failed to load JSON data: {str(e)}")


class NumPyDataSource(DataSource):
    """NumPy file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from NumPy file asynchronously."""
        try:
            # Use asyncio.to_thread for numpy operations
            data = await asyncio.to_thread(np.load, self.config.path)
            vectors = data[self.config.vector_field].tolist()
            
            metadata = None
            if self.config.metadata_fields:
                metadata = [{k: data[k].tolist() for k in self.config.metadata_fields}]
                
            return vectors, metadata
        except Exception as e:
            raise DataSourceError(f"Failed to load NumPy data: {str(e)}")


class HDF5DataSource(DataSource):
    """HDF5 file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from HDF5 file asynchronously."""
        try:
            # Use asyncio.to_thread for h5py operations
            def load_hdf5():
                with h5py.File(self.config.path, 'r') as f:
                    vectors = f[self.config.vector_field][:].tolist()
                    metadata = None
                    if self.config.metadata_fields:
                        metadata = [{k: f[k][:].tolist() for k in self.config.metadata_fields}]
                    return vectors, metadata
                    
            return await asyncio.to_thread(load_hdf5)
        except Exception as e:
            raise DataSourceError(f"Failed to load HDF5 data: {str(e)}")


class ParquetDataSource(DataSource):
    """Parquet file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from Parquet file asynchronously."""
        try:
            # Use asyncio.to_thread for pandas operations
            df = await asyncio.to_thread(pd.read_parquet, self.config.path)
            vectors = df[self.config.vector_field].tolist()
            
            metadata = None
            if self.config.metadata_fields:
                metadata = df[self.config.metadata_fields].to_dict('records')
                
            return vectors, metadata
        except Exception as e:
            raise DataSourceError(f"Failed to load Parquet data: {str(e)}")


class PickleDataSource(DataSource):
    """Pickle file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from Pickle file asynchronously."""
        try:
            async with aiofiles.open(self.config.path, 'rb') as f:
                content = await f.read()
                data = pickle.loads(content)
                
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
            raise DataSourceError(f"Failed to load Pickle data: {str(e)}")


class YAMLDataSource(DataSource):
    """YAML file data source."""
    
    async def load_data(self) -> Tuple[List[List[float]], Optional[List[Dict[str, Any]]]]:
        """Load data from YAML file asynchronously."""
        try:
            async with aiofiles.open(self.config.path) as f:
                content = await f.read()
                data = yaml.safe_load(content)
                
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
            raise DataSourceError(f"Failed to load YAML data: {str(e)}")


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


async def load_data_source(config_path: str) -> DataSource:
    """
    Load data source from configuration file asynchronously.
    
    Args:
        config_path: Path to the data source configuration file
        
    Returns:
        DataSource instance
        
    Raises:
        DataSourceError: If loading the data source fails
    """
    try:
        config_path = Path(config_path)
        
        # Load configuration from file
        if config_path.suffix in ('.yaml', '.yml'):
            async with aiofiles.open(config_path) as f:
                content = await f.read()
                config_data = yaml.safe_load(content)
        else:
            async with aiofiles.open(config_path) as f:
                content = await f.read()
                config_data = json.loads(content)
                
        # Create and validate configuration
        config = DataSourceConfig(**config_data)
        return DataSourceFactory.create(config)
    except Exception as e:
        raise DataSourceError(f"Failed to load data source: {str(e)}") 