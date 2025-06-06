"""
Advanced Milvus features including partitions, hybrid queries, and more with async support.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import asyncio
import logging
from datetime import datetime, timedelta
import json

from .client import AsyncMilvusClient
from .config import MilvusConfig
from .exceptions import AdvancedOperationError


class PartitionConfig(BaseModel):
    """Configuration for partition management."""
    partition_name: str = Field(..., description="Partition name")
    description: Optional[str] = Field(None, description="Partition description")
    tags: Optional[List[str]] = Field(None, description="Partition tags")


class HybridQueryConfig(BaseModel):
    """Configuration for hybrid queries."""
    vector_field: str = Field("vector", description="Vector field name")
    scalar_fields: List[str] = Field(..., description="Scalar fields for filtering")
    metric_type: str = Field("L2", description="Distance metric type")
    top_k: int = Field(10, description="Number of results to return")
    params: Optional[Dict[str, Any]] = Field(None, description="Search parameters")


class AdvancedMilvusClient(AsyncMilvusClient):
    """Advanced Milvus client with additional features."""
    
    def __init__(self, config: Union[str, MilvusConfig]):
        """Initialize the advanced client."""
        super().__init__(config)
        self.collection = self._get_collection()
        
    async def create_partition(self, partition_config: PartitionConfig) -> None:
        """Create a new partition asynchronously."""
        try:
            await self.collection.create_partition(
                partition_name=partition_config.partition_name,
                description=partition_config.description,
                tags=partition_config.tags
            )
        except Exception as e:
            raise Exception(f"Failed to create partition: {str(e)}")
            
    async def drop_partition(self, partition_name: str) -> None:
        """Drop a partition asynchronously."""
        try:
            await self.collection.drop_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to drop partition: {str(e)}")
            
    async def list_partitions(self) -> List[Dict[str, Any]]:
        """List all partitions asynchronously."""
        try:
            partitions = await self.collection.partitions
            return [
                {
                    "name": p.name,
                    "description": p.description,
                    "tags": p.tags,
                    "num_entities": p.num_entities
                }
                for p in partitions
            ]
        except Exception as e:
            raise Exception(f"Failed to list partitions: {str(e)}")
            
    async def hybrid_search(
        self,
        vectors: List[List[float]],
        query_config: HybridQueryConfig,
        partition_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with vector and scalar filtering asynchronously."""
        try:
            # Build search parameters
            search_params = {
                "metric_type": query_config.metric_type,
                "params": query_config.params or {"nprobe": 10}
            }
            
            # Build scalar filter expressions
            scalar_filters = []
            for field in query_config.scalar_fields:
                if field in kwargs:
                    scalar_filters.append(f"{field} == {kwargs[field]}")
                    
            # Combine filters
            filter_expr = " and ".join(scalar_filters) if scalar_filters else None
            
            # Perform search
            results = await self.collection.search(
                data=vectors,
                anns_field=query_config.vector_field,
                param=search_params,
                limit=query_config.top_k,
                expr=filter_expr,
                partition_names=partition_names,
                output_fields=query_config.scalar_fields
            )
            
            # Format results
            formatted_results = []
            for hits in results:
                hit_results = []
                for hit in hits:
                    result = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "metadata": hit.entity.get("metadata", {})
                    }
                    for field in query_config.scalar_fields:
                        result[field] = hit.entity.get(field)
                    hit_results.append(result)
                formatted_results.append(hit_results)
                
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Failed to perform hybrid search: {str(e)}")
            
    async def create_index(
        self,
        field_name: str,
        index_type: str,
        metric_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create an index on a field asynchronously."""
        try:
            await self.collection.create_index(
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params or {}
            )
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
            
    async def drop_index(self, field_name: str) -> None:
        """Drop an index from a field asynchronously."""
        try:
            await self.collection.drop_index(field_name)
        except Exception as e:
            raise Exception(f"Failed to drop index: {str(e)}")
            
    async def get_index_info(self, field_name: str) -> Dict[str, Any]:
        """Get index information for a field asynchronously."""
        try:
            index = await self.collection.index(field_name)
            return {
                "field_name": index.field_name,
                "index_type": index.index_type,
                "metric_type": index.metric_type,
                "params": index.params
            }
        except Exception as e:
            raise Exception(f"Failed to get index info: {str(e)}")
            
    async def load_partition(self, partition_name: str) -> None:
        """Load a partition into memory asynchronously."""
        try:
            await self.collection.load_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to load partition: {str(e)}")
            
    async def release_partition(self, partition_name: str) -> None:
        """Release a partition from memory asynchronously."""
        try:
            await self.collection.release_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to release partition: {str(e)}")
            
    async def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """Get statistics for a partition asynchronously."""
        try:
            partition = await self.collection.partition(partition_name)
            return {
                "name": partition.name,
                "description": partition.description,
                "tags": partition.tags,
                "num_entities": partition.num_entities,
                "is_loaded": partition.is_loaded
            }
        except Exception as e:
            raise Exception(f"Failed to get partition stats: {str(e)}")
            
    async def compact(self) -> None:
        """Compact the collection to remove deleted entities asynchronously."""
        try:
            await self.collection.compact()
        except Exception as e:
            raise Exception(f"Failed to compact collection: {str(e)}")
            
    async def get_compaction_state(self) -> Dict[str, Any]:
        """Get the current compaction state asynchronously."""
        try:
            state = await self.collection.get_compaction_state()
            return {
                "state": state.state,
                "executing_plans": state.executing_plans,
                "timeout_plans": state.timeout_plans,
                "completed_plans": state.completed_plans
            }
        except Exception as e:
            raise Exception(f"Failed to get compaction state: {str(e)}") 