"""
Cloud storage integration for various providers with async support.
"""

import os
import asyncio
import aiofiles
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3
from botocore.exceptions import ClientError
from google.cloud import storage
from google.oauth2 import service_account
import azure.storage.blob
from azure.identity import DefaultAzureCredential
import aiohttp
from pydantic import BaseModel, Field

from ai_prishtina_milvus_client.exceptions import CloudStorageError


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
    async def connect(self) -> None:
        """Establish connection to cloud storage asynchronously."""
        pass
        
    @abstractmethod
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from cloud storage asynchronously."""
        pass
        
    @abstractmethod
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to cloud storage asynchronously."""
        pass
        
    @abstractmethod
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in the bucket with optional prefix asynchronously."""
        pass


class S3Storage(CloudStorage):
    """AWS S3 storage implementation."""
    
    async def connect(self) -> None:
        """Connect to AWS S3 asynchronously."""
        try:
            if self.config.credentials:
                self._client = boto3.client(
                    's3',
                    aws_access_key_id=self.config.credentials.get('access_key_id'),
                    aws_secret_access_key=self.config.credentials.get('secret_access_key'),
                    region_name=self.config.region
                )
            else:
                self._client = boto3.client('s3', region_name=self.config.region)
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to S3: {str(e)}")
            
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from S3 asynchronously."""
        try:
            def download():
                self._client.download_file(self.config.bucket, remote_path, local_path)
            await asyncio.to_thread(download)
        except ClientError as e:
            raise CloudStorageError(f"Failed to download file from S3: {str(e)}")
            
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to S3 asynchronously."""
        try:
            def upload():
                self._client.upload_file(local_path, self.config.bucket, remote_path)
            await asyncio.to_thread(upload)
        except ClientError as e:
            raise CloudStorageError(f"Failed to upload file to S3: {str(e)}")
            
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in S3 bucket asynchronously."""
        try:
            prefix = prefix or self.config.prefix
            def list_objects():
                response = self._client.list_objects_v2(
                    Bucket=self.config.bucket,
                    Prefix=prefix
                )
                return [obj['Key'] for obj in response.get('Contents', [])]
            return await asyncio.to_thread(list_objects)
        except ClientError as e:
            raise CloudStorageError(f"Failed to list files in S3: {str(e)}")


class GCPStorage(CloudStorage):
    """Google Cloud Storage implementation."""
    
    async def connect(self) -> None:
        """Connect to Google Cloud Storage asynchronously."""
        try:
            if self.config.credentials:
                credentials = service_account.Credentials.from_service_account_info(
                    self.config.credentials
                )
                self._client = storage.Client(credentials=credentials)
            else:
                self._client = storage.Client()
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to GCS: {str(e)}")
            
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from GCS asynchronously."""
        try:
            def download():
                bucket = self._client.bucket(self.config.bucket)
                blob = bucket.blob(remote_path)
                blob.download_to_filename(local_path)
            await asyncio.to_thread(download)
        except Exception as e:
            raise CloudStorageError(f"Failed to download file from GCS: {str(e)}")
            
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to GCS asynchronously."""
        try:
            def upload():
                bucket = self._client.bucket(self.config.bucket)
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_path)
            await asyncio.to_thread(upload)
        except Exception as e:
            raise CloudStorageError(f"Failed to upload file to GCS: {str(e)}")
            
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in GCS bucket asynchronously."""
        try:
            prefix = prefix or self.config.prefix
            def list_blobs():
                bucket = self._client.bucket(self.config.bucket)
                blobs = bucket.list_blobs(prefix=prefix)
                return [blob.name for blob in blobs]
            return await asyncio.to_thread(list_blobs)
        except Exception as e:
            raise CloudStorageError(f"Failed to list files in GCS: {str(e)}")


class AzureStorage(CloudStorage):
    """Azure Blob Storage implementation."""
    
    async def connect(self) -> None:
        """Connect to Azure Blob Storage asynchronously."""
        try:
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
        except Exception as e:
            raise CloudStorageError(f"Failed to connect to Azure: {str(e)}")
            
    async def download_file(self, remote_path: str, local_path: str) -> None:
        """Download file from Azure Blob Storage asynchronously."""
        try:
            async def download():
                container_client = self._client.get_container_client(self.config.bucket)
                blob_client = container_client.get_blob_client(remote_path)
                async with aiofiles.open(local_path, "wb") as f:
                    await f.write(await blob_client.download_blob().readall())
            await download()
        except Exception as e:
            raise CloudStorageError(f"Failed to download file from Azure: {str(e)}")
            
    async def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload file to Azure Blob Storage asynchronously."""
        try:
            async def upload():
                container_client = self._client.get_container_client(self.config.bucket)
                blob_client = container_client.get_blob_client(remote_path)
                async with aiofiles.open(local_path, "rb") as f:
                    content = await f.read()
                    await blob_client.upload_blob(content, overwrite=True)
            await upload()
        except Exception as e:
            raise CloudStorageError(f"Failed to upload file to Azure: {str(e)}")
            
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files in Azure Blob Storage asynchronously."""
        try:
            prefix = prefix or self.config.prefix
            def list_blobs():
                container_client = self._client.get_container_client(self.config.bucket)
                blobs = container_client.list_blobs(name_starts_with=prefix)
                return [blob.name for blob in blobs]
            return await asyncio.to_thread(list_blobs)
        except Exception as e:
            raise CloudStorageError(f"Failed to list files in Azure: {str(e)}")


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


async def load_cloud_storage(config_path: str) -> CloudStorage:
    """
    Load cloud storage from configuration file asynchronously.
    
    Args:
        config_path: Path to the cloud storage configuration file
        
    Returns:
        CloudStorage instance
        
    Raises:
        CloudStorageError: If loading the cloud storage fails
    """
    try:
        async with aiofiles.open(config_path) as f:
            content = await f.read()
            config_data = json.loads(content)
        config = CloudStorageConfig(**config_data)
        return CloudStorageFactory.create(config)
    except Exception as e:
        raise CloudStorageError(f"Failed to load cloud storage: {str(e)}") 