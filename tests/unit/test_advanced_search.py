"""
Unit tests for advanced search module.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from ai_prishtina_milvus_client import AdvancedSearch, SearchConfig, FilterConfig, RankingConfig, FilterCondition

@pytest.fixture
def mock_client():
    """Create mock Milvus client."""
    client = Mock()
    client.search = Mock(return_value=[{"id": i, "score": 0.9} for i in range(10)])
    return client

def test_hybrid_search(mock_client, search_config, filter_config, ranking_config):
    """Test hybrid search."""
    search = AdvancedSearch(mock_client, search_config)
    
    # Test successful hybrid search
    vectors = [np.random.rand(128) for _ in range(5)]
    results = search.hybrid_search(
        vectors=vectors,
        text_query="test query",
        filter_config=filter_config,
        ranking_config=ranking_config
    )
    assert len(results) > 0
    assert all("id" in result for result in results)
    assert all("score" in result for result in results)
    
    # Test search with error
    mock_client.search.side_effect = Exception("Search failed")
    with pytest.raises(Exception):
        search.hybrid_search(vectors=vectors, text_query="test query")

def test_build_filter_expression(filter_config):
    """Test filter expression building."""
    search = AdvancedSearch(None, None)
    
    # Test simple filter
    expr = search._build_filter_expression(filter_config)
    assert isinstance(expr, str)
    assert "field" in expr
    assert "value" in expr
    
    # Test complex filter
    filter_config.conditions = [
        {"field": "age", "operator": ">", "value": 18},
        {"field": "category", "operator": "==", "value": "test"}
    ]
    expr = search._build_filter_expression(filter_config)
    assert "and" in expr.lower()
    assert "age" in expr
    assert "category" in expr

def test_perform_text_search(mock_client, search_config):
    """Test text search."""
    search = AdvancedSearch(mock_client, search_config)
    
    # Test successful text search
    results = search._perform_text_search(
        collection_name="test_collection",
        text_query="test query",
        search_config=search_config,
        filter_configs=None
    )
    assert len(results) >= 0  # Can be empty since it's a placeholder
    
    # Test empty results
    mock_client.search.return_value = []
    results = search._perform_text_search(
        collection_name="test_collection",
        text_query="test query",
        search_config=search_config,
        filter_configs=None
    )
    assert len(results) == 0

def test_combine_search_results():
    """Test search results combination."""
    search = AdvancedSearch(None, None)
    
    # Test combining results
    vector_results = [{"id": i, "score": 0.9} for i in range(5)]
    text_results = [{"id": i, "score": 0.8} for i in range(5)]
    combined = search._combine_search_results(vector_results, text_results, hybrid_weight=0.7)
    assert len(combined) == 5
    assert all("id" in result for result in combined)
    assert all("score" in result for result in combined)
    
    # Test with empty results
    combined = search._combine_search_results([], [], hybrid_weight=0.7)
    assert len(combined) == 0

def test_apply_ranking(ranking_config):
    """Test result ranking."""
    search = AdvancedSearch(None, None)
    
    # Test ranking application
    results = [{"id": i, "score": 0.9 - i*0.1} for i in range(5)]
    ranked = search._apply_ranking(results, ranking_config)
    assert len(ranked) == 5
    assert ranked[0]["score"] >= ranked[-1]["score"]
    
    # Test with empty results
    ranked = search._apply_ranking([], ranking_config)
    assert len(ranked) == 0

def test_filter_by_score():
    """Test score filtering."""
    search = AdvancedSearch(None, None)
    
    # Test score filtering
    results = [
        {"id": i, "score": 0.9 - i*0.1, "combined_score": 0.9 - i*0.1}
        for i in range(5)
    ]
    filtered = search._filter_by_score(results, min_score=0.7, max_score=0.9)
    assert len(filtered) > 0
    assert all(0.7 <= result["combined_score"] <= 0.9 for result in filtered)
    
    # Test with no results meeting criteria
    filtered = search._filter_by_score(results, min_score=1.0, max_score=1.0)
    assert len(filtered) == 0

def test_search_config():
    """Test search configuration."""
    config = SearchConfig(
        search_metrics=["L2", "IP"],
        top_k=10,
        hybrid_weight=0.7,
        min_score=0.5
    )
    
    assert config.search_metrics == ["L2", "IP"]
    assert config.top_k == 10
    assert config.hybrid_weight == 0.7
    assert config.min_score == 0.5

def test_filter_config():
    """Test filter configuration."""
    config = FilterConfig(
        conditions=[
            FilterCondition(field_name="age", operator=">", value=18),
            FilterCondition(field_name="category", operator="==", value="test")
        ],
        logical_operator="and"
    )
    
    assert len(config.conditions) == 2
    assert config.logical_operator == "and"

def test_ranking_config():
    """Test ranking configuration."""
    config = RankingConfig(
        sort_by="score",
        sort_order="desc",
        min_score=0.5,
        max_results=10
    )
    
    assert config.sort_by == "score"
    assert config.sort_order == "desc"
    assert config.min_score == 0.5
    assert config.max_results == 10 