"""Tests for API integrations."""

import pytest
import asyncio
from typing import List, Dict, Any
import aiohttp
from unittest.mock import AsyncMock, patch

from ai_prishtina_milvus_client.api_integrations import APIClient, APIConfig, APIIntegration
from ai_prishtina_milvus_client.config import MilvusConfig
from ai_prishtina_milvus_client.exceptions import APIError


@pytest.fixture
def api_config() -> APIConfig:
    """Create API configuration."""
    return APIConfig(
        base_url="http://test-api.com",
        api_key="test-key",
        timeout=30.0,
        max_retries=3,
        retry_delay=1.0,
        headers={"Content-Type": "application/json"},
        verify_ssl=True
    )


@pytest.fixture
async def api_client(api_config: APIConfig):
    """Create API client."""
    async with APIClient(api_config) as client:
        yield client


@pytest.fixture
async def api_integration(milvus_config: MilvusConfig, api_config: APIConfig):
    """Create API integration."""
    integration = APIIntegration(milvus_config, api_config)
    yield integration
    await integration.cleanup()


@pytest.mark.asyncio
async def test_api_client_request(api_client: APIClient):
    """Test API client request."""
    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = {"data": "test"}
    
    # Mock session
    mock_session = AsyncMock()
    mock_session.request.return_value.__aenter__.return_value = mock_response
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Make request
        response = await api_client.request(
            method="GET",
            endpoint="/test",
            params={"key": "value"}
        )
        
        # Verify request
        mock_session.request.assert_called_once_with(
            "GET",
            "http://test-api.com/test",
            params={"key": "value"},
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
            timeout=30.0,
            ssl=True
        )
        
        # Verify response
        assert response == {"data": "test"}


@pytest.mark.asyncio
async def test_api_client_retry(api_client: APIClient):
    """Test API client retry on failure."""
    # Mock response
    mock_response = AsyncMock()
    mock_response.status = 500
    
    # Mock session
    mock_session = AsyncMock()
    mock_session.request.return_value.__aenter__.return_value = mock_response
    
    with patch("aiohttp.ClientSession", return_value=mock_session):
        # Make request
        with pytest.raises(APIError):
            await api_client.request(
                method="GET",
                endpoint="/test"
            )
        
        # Verify retries
        assert mock_session.request.call_count == 4  # Initial + 3 retries


@pytest.mark.asyncio
async def test_api_integration_fetch_data(api_integration: APIIntegration):
    """Test API integration fetch data."""
    # Mock API response
    mock_data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3]},
        {"id": 2, "vector": [0.4, 0.5, 0.6]}
    ]
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_data
    
    with patch.object(api_integration, "client", mock_client):
        # Fetch data
        data = await api_integration.fetch_data("/vectors")
        
        # Verify request
        mock_client.get.assert_called_once_with("/vectors")
        
        # Verify response
        assert data == mock_data


@pytest.mark.asyncio
async def test_api_integration_index_data(
    api_integration: APIIntegration,
    test_collection: str
):
    """Test API integration index data."""
    # Mock API response
    mock_data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3]},
        {"id": 2, "vector": [0.4, 0.5, 0.6]}
    ]
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_data
    
    with patch.object(api_integration, "client", mock_client):
        # Index data
        result = await api_integration.index_data(
            collection_name=test_collection,
            data_endpoint="/vectors"
        )
        
        # Verify request
        mock_client.get.assert_called_once_with("/vectors")
        
        # Verify result
        assert result["indexed_count"] == 2
        assert result["failed_count"] == 0


@pytest.mark.asyncio
async def test_api_integration_search(
    api_integration: APIIntegration,
    test_collection: str,
    test_vectors: List[List[float]]
):
    """Test API integration search."""
    # Mock API response
    mock_results = [
        {"id": 1, "distance": 0.1},
        {"id": 2, "distance": 0.2}
    ]
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_results
    
    with patch.object(api_integration, "client", mock_client):
        # Search
        results = await api_integration.search(
            collection_name=test_collection,
            vectors=test_vectors[:2],
            search_endpoint="/search"
        )
        
        # Verify request
        mock_client.post.assert_called_once_with(
            "/search",
            json={"vectors": test_vectors[:2]}
        )
        
        # Verify results
        assert results == mock_results


@pytest.mark.asyncio
async def test_api_integration_enrich(
    api_integration: APIIntegration,
    test_collection: str
):
    """Test API integration enrich."""
    # Mock API response
    mock_data = [
        {"id": 1, "metadata": {"key": "value"}},
        {"id": 2, "metadata": {"key": "value"}}
    ]
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_data
    
    with patch.object(api_integration, "client", mock_client):
        # Enrich data
        result = await api_integration.enrich(
            collection_name=test_collection,
            enrich_endpoint="/enrich"
        )
        
        # Verify request
        mock_client.get.assert_called_once_with("/enrich")
        
        # Verify result
        assert result["enriched_count"] == 2
        assert result["failed_count"] == 0


@pytest.mark.asyncio
async def test_api_integration_sync(
    api_integration: APIIntegration,
    test_collection: str
):
    """Test API integration sync."""
    # Mock API response
    mock_data = [
        {"id": 1, "vector": [0.1, 0.2, 0.3]},
        {"id": 2, "vector": [0.4, 0.5, 0.6]}
    ]
    
    # Mock API client
    mock_client = AsyncMock()
    mock_client.get.return_value = mock_data
    
    with patch.object(api_integration, "client", mock_client):
        # Sync data
        result = await api_integration.sync(
            collection_name=test_collection,
            sync_endpoint="/sync"
        )
        
        # Verify request
        mock_client.get.assert_called_once_with("/sync")
        
        # Verify result
        assert result["synced_count"] == 2
        assert result["failed_count"] == 0


@pytest.mark.asyncio
async def test_api_integration_error_handling(api_integration: APIIntegration):
    """Test API integration error handling."""
    # Mock API client
    mock_client = AsyncMock()
    mock_client.get.side_effect = APIError("API error")
    
    with patch.object(api_integration, "client", mock_client):
        # Test error handling
        with pytest.raises(APIError):
            await api_integration.fetch_data("/test")


@pytest.mark.asyncio
async def test_api_integration_context_manager(
    milvus_config: MilvusConfig,
    api_config: APIConfig
):
    """Test API integration context manager."""
    async with APIIntegration(milvus_config, api_config) as integration:
        # Mock API client
        mock_client = AsyncMock()
        mock_client.get.return_value = [{"id": 1, "vector": [0.1, 0.2, 0.3]}]
        
        with patch.object(integration, "client", mock_client):
            # Test context manager
            data = await integration.fetch_data("/test")
            assert data == [{"id": 1, "vector": [0.1, 0.2, 0.3]}] 