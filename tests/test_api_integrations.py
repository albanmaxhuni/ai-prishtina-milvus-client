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


@patch("requests.Session")
def test_openai_get_vectors(mock_session, openai_config):
    """Test getting vectors from OpenAI."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]
    }
    mock_sess.post.return_value = mock_response
    
    # Create OpenAI client
    client = APIClientFactory.create(openai_config)
    
    # Test get_vectors
    vectors = client.get_vectors(["test query 1", "test query 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


@patch("requests.Session")
def test_huggingface_get_vectors(mock_session, huggingface_config):
    """Test getting vectors from HuggingFace."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]
    mock_sess.post.return_value = mock_response
    
    # Create HuggingFace client
    client = APIClientFactory.create(huggingface_config)
    
    # Test get_vectors
    vectors = client.get_vectors(["test query 1", "test query 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


@patch("requests.Session")
def test_cohere_get_vectors(mock_session, cohere_config):
    """Test getting vectors from Cohere."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    }
    mock_sess.post.return_value = mock_response
    
    # Create Cohere client
    client = APIClientFactory.create(cohere_config)
    
    # Test get_vectors
    vectors = client.get_vectors(["test query 1", "test query 2"])
    assert len(vectors) == 2
    assert len(vectors[0]) == 3
    assert len(vectors[1]) == 3


@patch("requests.Session")
def test_openai_get_metadata(mock_session, openai_config):
    """Test getting metadata from OpenAI."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "text": "Category: A\nScore: 0.8",
            },
            {
                "text": "Category: B\nScore: 0.9",
            },
        ]
    }
    mock_sess.post.return_value = mock_response
    
    # Create OpenAI client
    client = APIClientFactory.create(openai_config)
    
    # Test get_metadata
    metadata = client.get_metadata(["test query 1", "test query 2"])
    assert len(metadata) == 2
    assert "category" in metadata[0]
    assert "score" in metadata[0]


@patch("requests.Session")
def test_huggingface_get_metadata(mock_session, huggingface_config):
    """Test getting metadata from HuggingFace."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = [
        {"label": "A", "score": 0.8},
        {"label": "B", "score": 0.9},
    ]
    mock_sess.post.return_value = mock_response
    
    # Create HuggingFace client
    client = APIClientFactory.create(huggingface_config)
    
    # Test get_metadata
    metadata = client.get_metadata(["test query 1", "test query 2"])
    assert len(metadata) == 2
    assert "category" in metadata[0]
    assert "score" in metadata[0]


@patch("requests.Session")
def test_cohere_get_metadata(mock_session, cohere_config):
    """Test getting metadata from Cohere."""
    # Mock session
    mock_sess = MagicMock()
    mock_session.return_value = mock_sess
    
    # Mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "classifications": [
            {"prediction": "A", "confidence": 0.8},
            {"prediction": "B", "confidence": 0.9},
        ]
    }
    mock_sess.post.return_value = mock_response
    
    # Create Cohere client
    client = APIClientFactory.create(cohere_config)
    
    # Test get_metadata
    metadata = client.get_metadata(["test query 1", "test query 2"])
    assert len(metadata) == 2
    assert "category" in metadata[0]
    assert "score" in metadata[0]


def test_invalid_api_type():
    """Test invalid API type."""
    config = APIConfig(
        service="invalid",
        base_url="https://api.example.com",
        api_key="test-key",
    )
    
    with pytest.raises(ValueError):
        APIClientFactory.create(config)


def test_missing_api_key():
    """Test missing API key."""
    config = APIConfig(
        service="openai",
        base_url="https://api.openai.com/v1",
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