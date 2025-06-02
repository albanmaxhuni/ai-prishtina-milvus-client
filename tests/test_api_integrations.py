"""Unit tests for API integrations."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ai_prishtina_milvus_client.api_integrations import (
    APIConfig,
    APIClientFactory,
    load_api_client,
)


@pytest.fixture
def openai_config():
    """Create OpenAI configuration."""
    return APIConfig(
        service="openai",
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="text-embedding-ada-002",
        parameters={
            "max_tokens": 100,
            "temperature": 0.7,
        },
    )


@pytest.fixture
def huggingface_config():
    """Create HuggingFace configuration."""
    return APIConfig(
        service="huggingface",
        base_url="https://api-inference.huggingface.co",
        api_key="test-key",
        model="sentence-transformers/all-MiniLM-L6-v2",
        parameters={
            "max_length": 128,
            "truncation": True,
        },
    )


@pytest.fixture
def cohere_config():
    """Create Cohere configuration."""
    return APIConfig(
        service="cohere",
        base_url="https://api.cohere.ai/v1",
        api_key="test-key",
        model="embed-english-v2.0",
        parameters={
            "truncate": "NONE",
        },
    )


@pytest.fixture
def mock_requests():
    """Create mock requests session."""
    with patch("requests.Session") as mock_session:
        mock_sess = MagicMock()
        mock_session.return_value = mock_sess
        yield mock_sess


def test_openai_client(mock_requests):
    """Test OpenAI client."""
    config = {
        "service": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "model": "text-embedding-ada-002"
    }
    
    # Mock embedding response
    mock_requests.post.return_value.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]}
        ]
    }
    
    client = APIClientFactory.create(APIConfig(**config))
    vectors = client.get_vectors(["test1", "test2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    
    # Mock metadata response
    mock_requests.post.return_value.json.return_value = {
        "choices": [
            {"text": "category: A\nscore: 0.8"},
            {"text": "category: B\nscore: 0.9"}
        ]
    }
    
    metadata = client.get_metadata(["test1", "test2"])
    assert len(metadata) == 2
    assert metadata[0]["category"] == "A"
    assert metadata[0]["score"] == "0.8"


def test_missing_api_key():
    """Test missing API key."""
    config = {
        "service": "openai",
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-ada-002"
    }
    
    with pytest.raises(ValueError, match="API key is required"):
        APIClientFactory.create(APIConfig(**config))


def test_huggingface_client(mock_requests):
    """Test HuggingFace client."""
    config = {
        "service": "huggingface",
        "base_url": "https://api.huggingface.co",
        "api_key": "test-key",
        "model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Mock embedding response
    mock_requests.post.return_value.json.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    client = APIClientFactory.create(APIConfig(**config))
    vectors = client.get_vectors(["test1", "test2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    
    # Mock metadata response
    mock_requests.post.return_value.json.return_value = [
        {"label": "A", "score": 0.8},
        {"label": "B", "score": 0.9}
    ]
    
    metadata = client.get_metadata(["test1", "test2"])
    assert len(metadata) == 2
    assert metadata[0]["category"] == "A"
    assert metadata[0]["score"] == 0.8


def test_cohere_client(mock_requests):
    """Test Cohere client."""
    config = {
        "service": "cohere",
        "base_url": "https://api.cohere.ai",
        "api_key": "test-key",
        "model": "embed-english-v2.0"
    }
    
    # Mock embedding response
    mock_requests.post.return_value.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
    }
    
    client = APIClientFactory.create(APIConfig(**config))
    vectors = client.get_vectors(["test1", "test2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    
    # Mock metadata response
    mock_requests.post.return_value.json.return_value = {
        "classifications": [
            {"prediction": "A", "confidence": 0.8},
            {"prediction": "B", "confidence": 0.9}
        ]
    }
    
    metadata = client.get_metadata(["test1", "test2"])
    assert len(metadata) == 2
    assert metadata[0]["category"] == "A"
    assert metadata[0]["score"] == 0.8


def test_invalid_api_type():
    """Test invalid API type."""
    config = APIConfig(
        service="invalid",
        base_url="https://api.example.com",
        api_key="test-key",
    )
    
    with pytest.raises(ValueError):
        APIClientFactory.create(config)


def test_load_api_client():
    """Test loading API configuration."""
    # Create temporary config file
    config_data = {
        "service": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "model": "text-embedding-ada-002",
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.7,
        },
    }
    
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as temp_file:
        temp_file.write(str(config_data).encode())
        config_path = temp_file.name
    
    # Test loading
    with patch("requests.Session") as mock_session:
        mock_sess = MagicMock()
        mock_session.return_value = mock_sess
        
        client = load_api_client(config_path)
        assert client is not None
    
    # Clean up
    os.unlink(config_path) 