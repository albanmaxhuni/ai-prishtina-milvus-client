"""
Batch operations and monitoring utilities for Milvus with async support.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field

from .client import AsyncMilvusClient
from .data_validation import DataValidator, VectorValidationConfig
from .exceptions import BatchOperationError
from .config import MilvusConfig


class BatchConfig(BaseModel):
    """Configuration for batch operations."""
    batch_size: int = Field(1000, description="Batch size for operations")
    max_workers: int = Field(4, description="Maximum number of worker tasks")
    timeout: float = Field(30.0, description="Operation timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    use_transactions: bool = Field(True, description="Use transactions for batch operations")
    validate_before_insert: bool = Field(True, description="Validate data before insertion")


class BatchMetrics(BaseModel):
    """Metrics for batch operations."""
    total_items: int = Field(0, description="Total number of items processed")
    successful_items: int = Field(0, description="Number of successfully processed items")
    failed_items: int = Field(0, description="Number of failed items")
    total_time: float = Field(0.0, description="Total processing time in seconds")
    average_time_per_item: float = Field(0.0, description="Average processing time per item")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors encountered")


class BatchProcessor:
    """Batch processing utilities for Milvus."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        batch_config: Optional[BatchConfig] = None,
        client: Optional[AsyncMilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.batch_config = batch_config or BatchConfig()
        self.client = client or AsyncMilvusClient(milvus_config)
        
    async def batch_insert(
        self,
        collection_name: str,
        data: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Insert data in batches asynchronously."""
        try:
            collection = await self.client.get_collection(collection_name)
            
            # Split data into batches
            batches = [
                data[i:i + self.batch_config.batch_size]
                for i in range(0, len(data), self.batch_config.batch_size)
            ]
            
            if metadata:
                metadata_batches = [
                    metadata[i:i + self.batch_config.batch_size]
                    for i in range(0, len(metadata), self.batch_config.batch_size)
                ]
            else:
                metadata_batches = [None] * len(batches)
                
            # Process batches concurrently
            tasks = []
            for batch, meta_batch in zip(batches, metadata_batches):
                task = asyncio.create_task(
                    self._process_batch(collection, batch, meta_batch)
                )
                tasks.append(task)
                
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            raise BatchOperationError(f"Batch insert failed: {str(e)}")
            
    async def _process_batch(
        self,
        collection: Any,
        batch: List[Dict[str, Any]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Process a single batch asynchronously."""
        for attempt in range(self.batch_config.retry_attempts):
            try:
                if self.batch_config.use_transactions:
                    async with collection.start_transaction():
                        await collection.insert(batch, metadata)
                else:
                    await collection.insert(batch, metadata)
                return
            except Exception as e:
                if attempt == self.batch_config.retry_attempts - 1:
                    raise BatchOperationError(f"Failed to process batch: {str(e)}")
                await asyncio.sleep(self.batch_config.retry_delay)
                
    async def batch_delete(
        self,
        collection_name: str,
        expr: str
    ) -> None:
        """Delete data in batches asynchronously."""
        try:
            collection = await self.client.get_collection(collection_name)
            
            # Get IDs to delete
            results = await collection.query(expr=expr, output_fields=["id"])
            ids = [r["id"] for r in results]
            
            # Split IDs into batches
            id_batches = [
                ids[i:i + self.batch_config.batch_size]
                for i in range(0, len(ids), self.batch_config.batch_size)
            ]
            
            # Process batches concurrently
            tasks = []
            for id_batch in id_batches:
                task = asyncio.create_task(
                    self._delete_batch(collection, id_batch)
                )
                tasks.append(task)
                
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            raise BatchOperationError(f"Batch delete failed: {str(e)}")
            
    async def _delete_batch(self, collection: Any, ids: List[int]) -> None:
        """Delete a single batch asynchronously."""
        for attempt in range(self.batch_config.retry_attempts):
            try:
                if self.batch_config.use_transactions:
                    async with collection.start_transaction():
                        await collection.delete(f"id in {ids}")
                else:
                    await collection.delete(f"id in {ids}")
                return
            except Exception as e:
                if attempt == self.batch_config.retry_attempts - 1:
                    raise BatchOperationError(f"Failed to delete batch: {str(e)}")
                await asyncio.sleep(self.batch_config.retry_delay)
                
    async def batch_update(
        self,
        collection_name: str,
        expr: str,
        data: Dict[str, Any]
    ) -> None:
        """Update data in batches asynchronously."""
        try:
            collection = await self.client.get_collection(collection_name)
            
            # Get IDs to update
            results = await collection.query(expr=expr, output_fields=["id"])
            ids = [r["id"] for r in results]
            
            # Split IDs into batches
            id_batches = [
                ids[i:i + self.batch_config.batch_size]
                for i in range(0, len(ids), self.batch_config.batch_size)
            ]
            
            # Process batches concurrently
            tasks = []
            for id_batch in id_batches:
                task = asyncio.create_task(
                    self._update_batch(collection, id_batch, data)
                )
                tasks.append(task)
                
            # Wait for all batches to complete
            await asyncio.gather(*tasks)
            
        except Exception as e:
            raise BatchOperationError(f"Batch update failed: {str(e)}")
            
    async def _update_batch(
        self,
        collection: Any,
        ids: List[int],
        data: Dict[str, Any]
    ) -> None:
        """Update a single batch asynchronously."""
        for attempt in range(self.batch_config.retry_attempts):
            try:
                if self.batch_config.use_transactions:
                    async with collection.start_transaction():
                        await collection.upsert(
                            [{"id": id, **data} for id in ids]
                        )
                else:
                    await collection.upsert(
                        [{"id": id, **data} for id in ids]
                    )
                return
            except Exception as e:
                if attempt == self.batch_config.retry_attempts - 1:
                    raise BatchOperationError(f"Failed to update batch: {str(e)}")
                await asyncio.sleep(self.batch_config.retry_delay)
                
    async def batch_search(
        self,
        collection_name: str,
        vectors: List[List[float]],
        search_params: Dict[str, Any],
        output_fields: Optional[List[str]] = None
    ) -> List[List[Dict[str, Any]]]:
        """Search vectors in batches asynchronously."""
        try:
            collection = await self.client.get_collection(collection_name)
            
            # Split vectors into batches
            vector_batches = [
                vectors[i:i + self.batch_config.batch_size]
                for i in range(0, len(vectors), self.batch_config.batch_size)
            ]
            
            # Process batches concurrently
            tasks = []
            for vector_batch in vector_batches:
                task = asyncio.create_task(
                    self._search_batch(
                        collection,
                        vector_batch,
                        search_params,
                        output_fields
                    )
                )
                tasks.append(task)
                
            # Wait for all batches to complete
            results = await asyncio.gather(*tasks)
            
            # Combine results
            return [item for sublist in results for item in sublist]
            
        except Exception as e:
            raise BatchOperationError(f"Batch search failed: {str(e)}")
            
    async def _search_batch(
        self,
        collection: Any,
        vectors: List[List[float]],
        search_params: Dict[str, Any],
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search a single batch asynchronously."""
        for attempt in range(self.batch_config.retry_attempts):
            try:
                results = await collection.search(
                    vectors,
                    search_params,
                    output_fields=output_fields
                )
                return results
            except Exception as e:
                if attempt == self.batch_config.retry_attempts - 1:
                    raise BatchOperationError(f"Failed to search batch: {str(e)}")
                await asyncio.sleep(self.batch_config.retry_delay)
                
    async def batch_operation(
        self,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute a batch operation asynchronously."""
        try:
            # Create semaphore to limit concurrent operations
            semaphore = asyncio.Semaphore(self.batch_config.max_workers)
            
            async def _execute_with_semaphore():
                async with semaphore:
                    return await operation(*args, **kwargs)
                    
            # Execute operation with timeout
            return await asyncio.wait_for(
                _execute_with_semaphore(),
                timeout=self.batch_config.timeout
            )
            
        except asyncio.TimeoutError:
            raise BatchOperationError("Operation timed out")
        except Exception as e:
            raise BatchOperationError(f"Batch operation failed: {str(e)}") 