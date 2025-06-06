"""
Advanced search features for Milvus operations including hybrid search, filtering, and ranking with async support.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from pydantic import BaseModel, Field
import logging
from datetime import datetime
import asyncio

class SearchConfig(BaseModel):
    """Configuration for search operations."""
    metric_type: str = Field("L2", description="Distance metric type")
    top_k: int = Field(10, description="Number of results to return")
    nprobe: int = Field(10, description="Number of clusters to search")
    min_score: float = Field(0.0, description="Minimum score threshold")
    max_score: float = Field(1.0, description="Maximum score threshold")
    use_hybrid: bool = Field(False, description="Whether to use hybrid search")
    hybrid_weight: float = Field(0.5, description="Weight for hybrid search")
    search_metrics: List[str] = Field(default_factory=lambda: ["L2"], description="List of search metrics")

class FilterCondition(BaseModel):
    field_name: str = Field(..., description="Field to filter on")
    operator: str = Field(..., description="Filter operator")
    value: Any = Field(..., description="Filter value")

class FilterConfig(BaseModel):
    conditions: List[FilterCondition] = Field(default_factory=list, description="List of filter conditions")
    logical_operator: str = Field("and", description="Logical operator for combining filters")

class RankingConfig(BaseModel):
    sort_by: str = Field(..., description="Field to sort by")
    sort_order: str = Field("desc", description="Sort order (asc/desc)")
    min_score: float = Field(0.0, description="Minimum score threshold")
    max_results: int = Field(10, description="Maximum number of results")
    field_name: str = Field("score", description="Field to rank on")
    weight: float = Field(1.0, description="Weight for ranking")

class AdvancedSearch:
    """Advanced search features for Milvus."""
    
    def __init__(self, client: Any, search_config: Optional[SearchConfig] = None):
        """
        Initialize advanced search.
        
        Args:
            client: Milvus client instance
            search_config: Optional search configuration
        """
        self.client = client
        self.search_config = search_config or SearchConfig()
        self.logger = logging.getLogger(__name__)
        
    async def hybrid_search(
        self,
        collection_name: str,
        vector: List[float],
        text_query: str,
        search_config: SearchConfig,
        filter_configs: Optional[List[FilterConfig]] = None,
        ranking_configs: Optional[List[RankingConfig]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and text search asynchronously.
        
        Args:
            collection_name: Name of collection to search
            vector: Query vector
            text_query: Text query
            search_config: Search configuration
            filter_configs: Optional list of filter configurations
            ranking_configs: Optional list of ranking configurations
            
        Returns:
            List of search results
        """
        try:
            # Prepare search parameters
            search_params = {
                "metric_type": search_config.metric_type,
                "params": {"nprobe": search_config.nprobe}
            }
            
            # Prepare filter expression
            expr = await self._build_filter_expression(filter_configs) if filter_configs else None
            
            # Perform vector search
            vector_results = await self.client.search(
                collection_name=collection_name,
                data=[vector],
                anns_field="vector",
                param=search_params,
                limit=search_config.top_k,
                expr=expr
            )
            
            # Perform text search if hybrid search is enabled
            if search_config.use_hybrid:
                text_results = await self._perform_text_search(
                    collection_name,
                    text_query,
                    search_config,
                    filter_configs
                )
                
                # Combine results
                results = await self._combine_search_results(
                    vector_results,
                    text_results,
                    search_config.hybrid_weight
                )
            else:
                results = vector_results
                
            # Apply ranking if configured
            if ranking_configs:
                results = await self._apply_ranking(results, ranking_configs)
                
            # Filter by score
            results = await self._filter_by_score(
                results,
                search_config.min_score,
                search_config.max_score
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {str(e)}")
            raise
            
    async def _build_filter_expression(self, filter_configs: List[FilterConfig]) -> str:
        """
        Build filter expression from configurations asynchronously.
        
        Args:
            filter_configs: List of filter configurations
            
        Returns:
            Filter expression string
        """
        expressions = []
        
        for config in filter_configs:
            if config.operator == "in":
                values = [f"'{v}'" for v in config.value]
                expr = f"{config.field_name} in [{', '.join(values)}]"
            else:
                expr = f"{config.field_name} {config.operator} {config.value}"
                
            expressions.append(expr)
            
        if len(expressions) == 1:
            return expressions[0]
            
        logical_op = filter_configs[0].logical_operator or "and"
        return f" {logical_op} ".join(f"({expr})" for expr in expressions)
        
    async def _perform_text_search(
        self,
        collection_name: str,
        text_query: str,
        search_config: SearchConfig,
        filter_configs: Optional[List[FilterConfig]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform text search asynchronously.
        
        Args:
            collection_name: Name of collection to search
            text_query: Text query
            search_config: Search configuration
            filter_configs: Optional list of filter configurations
            
        Returns:
            List of search results
        """
        # This is a placeholder for text search implementation
        # In a real implementation, this would use a text search engine
        # or Milvus's text search capabilities
        return []
        
    async def _combine_search_results(
        self,
        vector_results: List[Dict[str, Any]],
        text_results: List[Dict[str, Any]],
        hybrid_weight: float
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and text search results asynchronously.
        
        Args:
            vector_results: Vector search results
            text_results: Text search results
            hybrid_weight: Weight for hybrid search
            
        Returns:
            Combined search results
        """
        # Handle empty results
        if not vector_results and not text_results:
            return []
            
        # Create result map
        result_map = {}
        
        # Add vector results
        for result in vector_results:
            id = result["id"]
            result_map[id] = {
                "id": id,
                "vector_score": result["score"],
                "text_score": 0.0,
                "combined_score": result["score"] * (1 - hybrid_weight)
            }
            
        # Add text results
        for result in text_results:
            id = result["id"]
            if id in result_map:
                result_map[id]["text_score"] = result["score"]
                result_map[id]["combined_score"] += result["score"] * hybrid_weight
            else:
                result_map[id] = {
                    "id": id,
                    "vector_score": 0.0,
                    "text_score": result["score"],
                    "combined_score": result["score"] * hybrid_weight
                }
                
        # Convert to list and sort by combined score
        results = list(result_map.values())
        results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add 'score' key for compatibility with tests
        for r in results:
            r["score"] = r["combined_score"]
        return results
        
    async def _apply_ranking(
        self,
        results: List[Dict[str, Any]],
        ranking_configs: List[RankingConfig]
    ) -> List[Dict[str, Any]]:
        """
        Apply ranking to search results asynchronously.
        
        Args:
            results: Search results
            ranking_configs: List of ranking configurations
            
        Returns:
            Ranked search results
        """
        for result in results:
            ranking_score = 0.0
            
            for config in ranking_configs:
                field_value = result.get(config.field_name, 0.0)
                if isinstance(field_value, (int, float)):
                    ranking_score += field_value * config.weight
                    
            result["ranking_score"] = ranking_score
            
        # Sort by ranking score
        results.sort(key=lambda x: x["ranking_score"], reverse=(ranking_configs[0].sort_order == "desc"))
        
        return results
        
    async def _filter_by_score(
        self,
        results: List[Dict[str, Any]],
        min_score: float,
        max_score: float
    ) -> List[Dict[str, Any]]:
        """
        Filter results by score asynchronously.
        
        Args:
            results: Search results
            min_score: Minimum score threshold
            max_score: Maximum score threshold
            
        Returns:
            Filtered search results
        """
        return [
            result for result in results
            if min_score <= result["combined_score"] <= max_score
        ]
        
    async def search_by_date_range(
        self,
        collection_name: str,
        start_date: datetime,
        end_date: datetime,
        date_field: str = "created_at"
    ) -> List[Dict[str, Any]]:
        """
        Search by date range asynchronously.
        
        Args:
            collection_name: Name of collection to search
            start_date: Start date
            end_date: End date
            date_field: Name of date field
            
        Returns:
            List of search results
        """
        expr = f"{date_field} >= '{start_date.isoformat()}' and {date_field} <= '{end_date.isoformat()}'"
        
        return await self.client.query(
            collection_name=collection_name,
            expr=expr
        )
        
    async def search_by_metadata(
        self,
        collection_name: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Search by metadata asynchronously.
        
        Args:
            collection_name: Name of collection to search
            metadata: Metadata to search for
            
        Returns:
            List of search results
        """
        expressions = []
        
        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                values = [f"'{v}'" for v in value]
                expr = f"metadata['{key}'] in [{', '.join(values)}]"
            else:
                expr = f"metadata['{key}'] == '{value}'"
            expressions.append(expr)
            
        expr = " and ".join(f"({e})" for e in expressions)
        
        return await self.client.query(
            collection_name=collection_name,
            expr=expr
        ) 