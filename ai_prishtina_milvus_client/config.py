"""
Configuration management for the Milvus client.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class MilvusConfig(BaseModel):
    """Configuration model for Milvus connection and collection settings."""
    
    host: str = Field(default="localhost", description="Milvus server host")
    port: int = Field(default=19530, description="Milvus server port")
    user: Optional[str] = Field(default=None, description="Milvus username")
    password: Optional[str] = Field(default=None, description="Milvus password")
    db_name: str = Field(default="default", description="Database name")
    collection_name: str = Field(..., description="Collection name")
    dim: int = Field(..., description="Vector dimension")
    index_type: str = Field(default="IVF_FLAT", description="Index type")
    metric_type: str = Field(default="L2", description="Distance metric type")
    nlist: int = Field(default=1024, description="Number of clusters for IVF index")
    metadata_fields: Optional[list] = Field(default=None, description="List of metadata fields for the collection schema. Each field should be a dict with 'name' and 'type'.")
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "MilvusConfig":
        """
        Load configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            MilvusConfig instance
            
        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            
        if "milvus" not in config_data:
            raise ValueError("Configuration file must contain a 'milvus' section")
            
        return cls(**config_data["milvus"])
    
    def to_yaml(self, config_path: str) -> None:
        """
        Save configuration to a YAML file.
        
        Args:
            config_path: Path to save the YAML configuration file
        """
        config_path = Path(config_path)
        config_data = {"milvus": self.model_dump()}
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False) 