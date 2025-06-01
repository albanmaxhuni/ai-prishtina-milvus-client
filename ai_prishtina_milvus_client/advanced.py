"""
Advanced Milvus features including partitions, hybrid queries, and more.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

from .client import MilvusClient
from .config import MilvusConfig


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


class AdvancedMilvusClient(MilvusClient):
    """Advanced Milvus client with additional features."""
    
    def create_partition(self, partition_config: PartitionConfig) -> None:
        """Create a new partition."""
        try:
            self.collection.create_partition(
                partition_name=partition_config.partition_name,
                description=partition_config.description,
                tags=partition_config.tags
            )
        except Exception as e:
            raise Exception(f"Failed to create partition: {str(e)}")
            
    def drop_partition(self, partition_name: str) -> None:
        """Drop a partition."""
        try:
            self.collection.drop_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to drop partition: {str(e)}")
            
    def list_partitions(self) -> List[Dict[str, Any]]:
        """List all partitions."""
        try:
            partitions = self.collection.partitions
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
            
    def hybrid_search(
        self,
        vectors: List[List[float]],
        query_config: HybridQueryConfig,
        partition_names: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search with vector and scalar filtering."""
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
            results = self.collection.search(
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
            
    def create_index(
        self,
        field_name: str,
        index_type: str,
        metric_type: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create an index on a field."""
        try:
            self.collection.create_index(
                field_name=field_name,
                index_type=index_type,
                metric_type=metric_type,
                params=params or {}
            )
        except Exception as e:
            raise Exception(f"Failed to create index: {str(e)}")
            
    def drop_index(self, field_name: str) -> None:
        """Drop an index from a field."""
        try:
            self.collection.drop_index(field_name)
        except Exception as e:
            raise Exception(f"Failed to drop index: {str(e)}")
            
    def get_index_info(self, field_name: str) -> Dict[str, Any]:
        """Get index information for a field."""
        try:
            index = self.collection.index(field_name)
            return {
                "field_name": index.field_name,
                "index_type": index.index_type,
                "metric_type": index.metric_type,
                "params": index.params
            }
        except Exception as e:
            raise Exception(f"Failed to get index info: {str(e)}")
            
    def load_partition(self, partition_name: str) -> None:
        """Load a partition into memory."""
        try:
            self.collection.load_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to load partition: {str(e)}")
            
    def release_partition(self, partition_name: str) -> None:
        """Release a partition from memory."""
        try:
            self.collection.release_partition(partition_name)
        except Exception as e:
            raise Exception(f"Failed to release partition: {str(e)}")
            
    def get_partition_stats(self, partition_name: str) -> Dict[str, Any]:
        """Get statistics for a partition."""
        try:
            partition = self.collection.partition(partition_name)
            return {
                "name": partition.name,
                "description": partition.description,
                "tags": partition.tags,
                "num_entities": partition.num_entities,
                "is_loaded": partition.is_loaded
            }
        except Exception as e:
            raise Exception(f"Failed to get partition stats: {str(e)}")
            
    def compact(self) -> None:
        """Compact the collection to remove deleted entities."""
        try:
            self.collection.compact()
        except Exception as e:
            raise Exception(f"Failed to compact collection: {str(e)}")
            
    def get_compaction_state(self) -> Dict[str, Any]:
        """Get the current compaction state."""
        try:
            state = self.collection.get_compaction_state()
            return {
                "state": state.state,
                "executing_plans": state.executing_plans,
                "timeout_plans": state.timeout_plans,
                "completed_plans": state.completed_plans
            }
        except Exception as e:
            raise Exception(f"Failed to get compaction state: {str(e)}") 