"""
API integrations for vector embeddings and metadata.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import requests
import yaml
from google.cloud import aiplatform
from google.oauth2 import service_account
import boto3
from pydantic import BaseModel, Field


class APIConfig(BaseModel):
    """Configuration for API clients."""
    service: str = Field(..., description="API service name")
    base_url: str = Field(..., description="Base URL for API")
    api_key: Optional[str] = Field(None, description="API key")
    model: Optional[str] = Field(None, description="Model name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Service-specific parameters")
    credentials_path: Optional[str] = Field(None, description="Path to credentials file (for GCP/AWS)")
    region: Optional[str] = Field(None, description="AWS region (for Bedrock)")


class APIClient(ABC):
    """Abstract base class for API clients."""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = requests.Session()
        if self.config.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.config.api_key}"})
            
    @abstractmethod
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get vector embeddings for texts."""
        pass
        
    @abstractmethod
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata for texts."""
        pass


class OpenAIClient(APIClient):
    """OpenAI API client."""
    
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from OpenAI."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/embeddings",
            json={
                "input": texts,
                "model": self.config.model or "text-embedding-ada-002",
                **params
            }
        )
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
        
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from OpenAI."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/completions",
            json={
                "prompt": texts,
                "model": self.config.model or "text-davinci-003",
                **params
            }
        )
        response.raise_for_status()
        results = []
        for choice in response.json()["choices"]:
            text = choice["text"].strip()
            lines = text.split("\n")
            metadata = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip().lower()] = value.strip()
            results.append(metadata)
        return results


class HuggingFaceClient(APIClient):
    """HuggingFace API client."""
    
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from HuggingFace."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/models/{self.config.model}",
            json={
                "inputs": texts,
                **params
            }
        )
        response.raise_for_status()
        return response.json()
        
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from HuggingFace."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/models/{self.config.model}",
            json={
                "inputs": texts,
                **params
            }
        )
        response.raise_for_status()
        results = []
        for item in response.json():
            results.append({
                "category": item["label"],
                "score": item["score"]
            })
        return results


class CohereClient(APIClient):
    """Cohere API client."""
    
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from Cohere."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/embed",
            json={
                "texts": texts,
                "model": self.config.model or "embed-english-v2.0",
                **params
            }
        )
        response.raise_for_status()
        return response.json()["embeddings"]
        
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from Cohere."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.session.post(
            f"{self.config.base_url}/classify",
            json={
                "texts": texts,
                "model": self.config.model or "large",
                **params
            }
        )
        response.raise_for_status()
        results = []
        for item in response.json()["classifications"]:
            results.append({
                "category": item["prediction"],
                "score": item["confidence"]
            })
        return results


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
        
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from Vertex AI."""
        embeddings = self.model.get_embeddings(texts)
        return [embedding.values for embedding in embeddings]
        
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from Vertex AI."""
        # Use text classification model for metadata
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
        
    def get_vectors(self, texts: List[str], **kwargs) -> List[List[float]]:
        """Get embeddings from AWS Bedrock."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.client.invoke_model(
            modelId=self.config.model or "amazon.titan-embed-text-v1",
            body=json.dumps({
                "inputText": texts,
                **params
            })
        )
        result = json.loads(response["body"].read())
        return result["embeddings"]
        
    def get_metadata(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Get metadata from AWS Bedrock."""
        params = dict(self.config.parameters or {}, **kwargs)
        response = self.client.invoke_model(
            modelId=self.config.model or "anthropic.claude-v2",
            body=json.dumps({
                "prompt": texts,
                **params
            })
        )
        result = json.loads(response["body"].read())
        return [{"text": completion} for completion in result["completions"]]


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
        """Create an API client instance."""
        client_class = cls._clients.get(config.service.lower())
        if not client_class:
            raise ValueError(f"Unsupported API service: {config.service}")
        return client_class(config)


def load_api_client(config_path: str) -> APIClient:
    """Load API client from configuration file."""
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    config = APIConfig(**config_data)
    return APIClientFactory.create(config) 