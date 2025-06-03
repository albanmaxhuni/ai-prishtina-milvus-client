"""
Cloud storage integration for various providers.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError
from google.cloud import storage
from google.oauth2 import service_account
import azure.storage.blob
from azure.identity import DefaultAzureCredential
import requests
from pydantic import BaseModel, Field


class CloudStorageConfig(BaseModel):
    """Base configuration for cloud storage."""
    provider: str = Field(..., description="Cloud provider (aws, gcp, azure)")
    bucket: str = Field(..., description="Bucket/container name")
    prefix: Optional[str] = Field(None, description="Path prefix in bucket")
    region: Optional[str] = Field(None, description="Region for the bucket")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Provider-specific credentials")


class CloudStorage(ABC):
    """Abstract base class for cloud storage providers."""
    
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self._client = None
        
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to cloud storage."""
        pass
        
    @abstractmethod
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from cloud storage."""
        pass
        
    @abstractmethod
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to cloud storage."""
        pass
        
    @abstractmethod
    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in the bucket with optional prefix."""
        pass


class S3Storage(CloudStorage):
    """AWS S3 storage implementation."""
    
    def connect(self) -> None:
        """Connect to AWS S3."""
        if self.config.credentials:
            self._client = boto3.client(
                's3',
                aws_access_key_id=self.config.credentials.get('access_key_id'),
                aws_secret_access_key=self.config.credentials.get('secret_access_key'),
                region_name=self.config.region
            )
        else:
            self._client = boto3.client('s3', region_name=self.config.region)
            
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from S3."""
        try:
            self._client.download_file(self.config.bucket, remote_path, local_path)
        except ClientError as e:
            raise Exception(f"Failed to download file from S3: {str(e)}")
            
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to S3."""
        try:
            self._client.upload_file(local_path, self.config.bucket, remote_path)
        except ClientError as e:
            raise Exception(f"Failed to upload file to S3: {str(e)}")
            
    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in S3 bucket."""
        try:
            prefix = prefix or self.config.prefix
            response = self._client.list_objects_v2(
                Bucket=self.config.bucket,
                Prefix=prefix
            )
            return [obj['Key'] for obj in response.get('Contents', [])]
        except ClientError as e:
            raise Exception(f"Failed to list files in S3: {str(e)}")


class GCPStorage(CloudStorage):
    """Google Cloud Storage implementation."""
    
    def connect(self) -> None:
        """Connect to Google Cloud Storage."""
        if self.config.credentials:
            credentials = service_account.Credentials.from_service_account_info(
                self.config.credentials
            )
            self._client = storage.Client(credentials=credentials)
        else:
            self._client = storage.Client()
            
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from GCS."""
        try:
            bucket = self._client.bucket(self.config.bucket)
            blob = bucket.blob(remote_path)
            blob.download_to_filename(local_path)
        except Exception as e:
            raise Exception(f"Failed to download file from GCS: {str(e)}")
            
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to GCS."""
        try:
            bucket = self._client.bucket(self.config.bucket)
            blob = bucket.blob(remote_path)
            blob.upload_from_filename(local_path)
        except Exception as e:
            raise Exception(f"Failed to upload file to GCS: {str(e)}")
            
    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in GCS bucket."""
        try:
            prefix = prefix or self.config.prefix
            bucket = self._client.bucket(self.config.bucket)
            blobs = bucket.list_blobs(prefix=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            raise Exception(f"Failed to list files in GCS: {str(e)}")


class AzureStorage(CloudStorage):
    """Azure Blob Storage implementation."""
    
    def connect(self) -> None:
        """Connect to Azure Blob Storage."""
        if self.config.credentials:
            self._client = azure.storage.blob.BlobServiceClient(
                account_url=f"https://{self.config.credentials['account_name']}.blob.core.windows.net",
                credential=self.config.credentials['account_key']
            )
        else:
            credential = DefaultAzureCredential()
            self._client = azure.storage.blob.BlobServiceClient(
                account_url=f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net",
                credential=credential
            )
            
    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from Azure Blob Storage."""
        try:
            container_client = self._client.get_container_client(self.config.bucket)
            blob_client = container_client.get_blob_client(remote_path)
            with open(local_path, "wb") as f:
                blob_client.download_blob().readinto(f)
        except Exception as e:
            raise Exception(f"Failed to download file from Azure: {str(e)}")
            
    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to Azure Blob Storage."""
        try:
            container_client = self._client.get_container_client(self.config.bucket)
            blob_client = container_client.get_blob_client(remote_path)
            with open(local_path, "rb") as f:
                blob_client.upload_blob(f.read(), overwrite=True)
        except Exception as e:
            raise Exception(f"Failed to upload file to Azure: {str(e)}")
            
    def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in Azure Blob Storage."""
        try:
            prefix = prefix or self.config.prefix
            container_client = self._client.get_container_client(self.config.bucket)
            blobs = container_client.list_blobs(name_starts_with=prefix)
            return [blob.name for blob in blobs]
        except Exception as e:
            raise Exception(f"Failed to list files in Azure: {str(e)}")


class CloudStorageFactory:
    """Factory for creating cloud storage instances."""
    
    _providers = {
        "aws": S3Storage,
        "gcp": GCPStorage,
        "azure": AzureStorage,
    }
    
    @classmethod
    def create(cls, config: CloudStorageConfig) -> CloudStorage:
        """
        Create a cloud storage instance.
        
        Args:
            config: Cloud storage configuration
            
        Returns:
            CloudStorage instance
            
        Raises:
            ValueError: If provider is not supported
        """
        provider_class = cls._providers.get(config.provider.lower())
        if not provider_class:
            raise ValueError(f"Unsupported cloud provider: {config.provider}")
        return provider_class(config)


def load_cloud_storage(config_path: str) -> CloudStorage:
    """
    Load cloud storage from configuration file.
    
    Args:
        config_path: Path to the cloud storage configuration file
        
    Returns:
        CloudStorage instance
    """
    import json
    with open(config_path) as f:
        config_data = json.load(f)
    config = CloudStorageConfig(**config_data)
    return CloudStorageFactory.create(config) 