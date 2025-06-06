"""
API integrations for vector embeddings and metadata with async support.
"""

import json
import os
import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Awaitable

import aiohttp
import yaml
from google.cloud import aiplatform
from google.oauth2 import service_account
import boto3
from pydantic import BaseModel, Field

from ai_prishtina_milvus_client.exceptions import APIError


class APIConfig(BaseModel):
    """Configuration for API clients."""
    service: str = Field(..., description="API service name")
    base_url: str = Field(..., description="Base URL for API")
    api_key: Optional[str] = Field(None, description="API key")
    model: Optional[str] = Field(None, description="Model name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Service-specific parameters")
    credentials_path: Optional[str] = Field(None, description="Path to credentials file (for GCP/AWS)")
    region: Optional[str] = Field(None, description="AWS region (for Bedrock)")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    verify_ssl: bool = Field(True, description="Verify SSL certificates")


class APIClient(ABC):
    """Abstract base class for API clients."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            headers={
                "Content-Type": "application/json",
                **self.config.headers
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        if self.config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.config.api_key}"})
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    @abstractmethod
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get vector embeddings for texts asynchronously."""
        pass
        
    @abstractmethod
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata for texts asynchronously."""
        pass


class OpenAIClient(APIClient):
    """OpenAI API client."""
    
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from OpenAI asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/embeddings",
                json={
                    "input": texts,
                    "model": self.config.model or "text-embedding-ada-002",
                    **params
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return [item["embedding"] for item in data["data"]]
        except Exception as e:
            raise APIError(f"Failed to get vectors from OpenAI: {str(e)}")
        
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from OpenAI asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/completions",
                json={
                    "prompt": texts,
                    "model": self.config.model or "text-davinci-003",
                    **params
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                results = []
                for choice in data["choices"]:
                    text = choice["text"].strip()
                    lines = text.split("\n")
                    metadata = {}
                    for line in lines:
                        if ":" in line:
                            key, value = line.split(":", 1)
                            metadata[key.strip().lower()] = value.strip()
                    results.append(metadata)
                return results
        except Exception as e:
            raise APIError(f"Failed to get metadata from OpenAI: {str(e)}")


class HuggingFaceClient(APIClient):
    """HuggingFace API client."""
    
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from HuggingFace asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/models/{self.config.model}",
                json={
                    "inputs": texts,
                    **params
                }
            ) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            raise APIError(f"Failed to get vectors from HuggingFace: {str(e)}")
        
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from HuggingFace asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/models/{self.config.model}",
                json={
                    "inputs": texts,
                    **params
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return [{
                    "category": item["label"],
                    "score": item["score"]
                } for item in data]
        except Exception as e:
            raise APIError(f"Failed to get metadata from HuggingFace: {str(e)}")


class CohereClient(APIClient):
    """Cohere API client."""
    
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from Cohere asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/embed",
                json={
                    "texts": texts,
                    "model": self.config.model or "embed-english-v2.0",
                    **params
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return data["embeddings"]
        except Exception as e:
            raise APIError(f"Failed to get vectors from Cohere: {str(e)}")
        
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from Cohere asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            async with self.session.post(
                f"{self.config.base_url}/classify",
                json={
                    "texts": texts,
                    "model": self.config.model or "large",
                    **params
                }
            ) as response:
                response.raise_for_status()
                data = await response.json()
                return [{
                    "category": item["prediction"],
                    "score": item["confidence"]
                } for item in data["classifications"]]
        except Exception as e:
            raise APIError(f"Failed to get metadata from Cohere: {str(e)}")


class GoogleVertexAIClient(APIClient):
    """Google Vertex AI client."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        credentials = service_account.Credentials.from_service_account_file(
            config.credentials_path
        ) if config.credentials_path else None
        self.client = aiplatform.init(
            project=config.parameters.get("project_id"),
            location=config.parameters.get("location", "us-central1"),
            credentials=credentials
        )
        self.model = aiplatform.TextEmbeddingModel.from_pretrained(
            config.model or "textembedding-gecko@001"
        )
        
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from Vertex AI asynchronously."""
        try:
            def get_embeddings():
                embeddings = self.model.get_embeddings(texts)
                return [embedding.values for embedding in embeddings]
            return await asyncio.to_thread(get_embeddings)
        except Exception as e:
            raise APIError(f"Failed to get vectors from Vertex AI: {str(e)}")
        
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from Vertex AI asynchronously."""
        try:
            def get_metadata():
                model = aiplatform.TextClassificationModel.from_pretrained(
                    self.config.parameters.get("classification_model", "text-bison@001")
                )
                predictions = model.predict(texts)
                return [
                    {
                        "label": pred.label,
                        "confidence": pred.confidence
                    }
                    for pred in predictions
                ]
            return await asyncio.to_thread(get_metadata)
        except Exception as e:
            raise APIError(f"Failed to get metadata from Vertex AI: {str(e)}")


class AWSBedrockClient(APIClient):
    """AWS Bedrock client."""
    
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=config.region or "us-east-1",
            aws_access_key_id=config.parameters.get("aws_access_key_id"),
            aws_secret_access_key=config.parameters.get("aws_secret_access_key")
        )
        
    async def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from AWS Bedrock asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            def get_embeddings():
                response = self.client.invoke_model(
                    modelId=self.config.model or "amazon.titan-embed-text-v1",
                    body=json.dumps({
                        "inputText": texts,
                        **params
                    })
                )
                result = json.loads(response["body"].read())
                return result["embeddings"]
            return await asyncio.to_thread(get_embeddings)
        except Exception as e:
            raise APIError(f"Failed to get vectors from AWS Bedrock: {str(e)}")
        
    async def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from AWS Bedrock asynchronously."""
        try:
            params = dict(self.config.parameters or {}, **kwargs)
            def get_metadata():
                response = self.client.invoke_model(
                    modelId=self.config.model or "anthropic.claude-v2",
                    body=json.dumps({
                        "prompt": texts,
                        **params
                    })
                )
                result = json.loads(response["body"].read())
                return [{"text": completion} for completion in result["completions"]]
            return await asyncio.to_thread(get_metadata)
        except Exception as e:
            raise APIError(f"Failed to get metadata from AWS Bedrock: {str(e)}")


class APIClientFactory:
    """Factory for creating API clients."""
    
    _clients = {
        "openai": OpenAIClient,
        "huggingface": HuggingFaceClient,
        "cohere": CohereClient,
        "vertex": GoogleVertexAIClient,
        "bedrock": AWSBedrockClient,
    }
    
    @classmethod
    def create(cls, config: APIConfig) -> APIClient:
        """
        Create an API client instance.
        
        Args:
            config: API configuration
            
        Returns:
            APIClient instance
            
        Raises:
            ValueError: If service is not supported
        """
        client_class = cls._clients.get(config.service.lower())
        if not client_class:
            raise ValueError(f"Unsupported API service: {config.service}")
        return client_class(config)


async def load_api_client(config_path: str) -> APIClient:
    """
    Load API client from configuration file asynchronously.
    
    Args:
        config_path: Path to the API configuration file
        
    Returns:
        APIClient instance
        
    Raises:
        APIError: If loading the API client fails
    """
    try:
        async with aiofiles.open(config_path) as f:
            content = await f.read()
            config_data = yaml.safe_load(content)
        config = APIConfig(**config_data)
        return APIClientFactory.create(config)
    except Exception as e:
        raise APIError(f"Failed to load API client: {str(e)}") 