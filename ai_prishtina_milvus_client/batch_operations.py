"""
Batch operations and monitoring utilities for Milvus.
"""

from typing import List, Dict, Any, Optional, Callable, Union
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel, Field

from .client import MilvusClient
from .data_validation import DataValidator, VectorValidationConfig

class BatchConfig(BaseModel):
    """Configuration for batch operations."""
    batch_size: int = Field(1000, description="Number of items per batch")
    max_workers: int = Field(4, description="Maximum number of worker threads")
    progress_display: bool = Field(True, description="Whether to show progress bar")
    timeout: int = Field(30, description="Batch operation timeout in seconds")
    validation_config: Optional[VectorValidationConfig] = Field(None, description="Vector validation config")

class BatchMetrics(BaseModel):
    """Metrics for batch operations."""
    total_items: int = Field(0, description="Total number of items processed")
    successful_items: int = Field(0, description="Number of successfully processed items")
    failed_items: int = Field(0, description="Number of failed items")
    total_time: float = Field(0.0, description="Total processing time in seconds")
    average_time_per_item: float = Field(0.0, description="Average processing time per item")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors encountered")

class BatchProcessor:
    """Utility class for batch operations."""
    
    def __init__(self, client: MilvusClient, config: BatchConfig):
        """
        Initialize batch processor.
        
        Args:
            client: Milvus client instance
            config: Batch configuration
        """
        self.client = client
        self.config = config
        self.validator = DataValidator()
        
    def batch_insert(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        callback: Optional[Callable[[BatchMetrics], None]] = None
    ) -> BatchMetrics:
        """
        Insert vectors in batches with progress tracking.
        
        Args:
            vectors: List of vectors to insert
            metadata: Optional list of metadata dictionaries
            callback: Optional callback function for metrics
            
        Returns:
            Batch metrics
        """
        start_time = time.time()
        metrics = BatchMetrics(total_items=len(vectors))
        
        # Validate vectors if config provided
        if self.config.validation_config:
            vectors = self.validator.validate_vectors(vectors, self.config.validation_config)
            
        # Create batches
        batches = [
            (vectors[i:i + self.config.batch_size],
             metadata[i:i + self.config.batch_size] if metadata else None)
            for i in range(0, len(vectors), self.config.batch_size)
        ]
        
        # Process batches
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for batch_vectors, batch_metadata in batches:
                future = executor.submit(
                    self._process_batch,
                    batch_vectors,
                    batch_metadata
                )
                futures.append(future)
                
            # Monitor progress
            if self.config.progress_display:
                with tqdm(total=len(vectors), desc="Inserting vectors") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            metrics.successful_items += result["successful"]
                            metrics.failed_items += result["failed"]
                            metrics.errors.extend(result["errors"])
                            pbar.update(result["successful"] + result["failed"])
                        except Exception as e:
                            metrics.failed_items += len(batch_vectors)
                            metrics.errors.append({"error": str(e)})
                            pbar.update(len(batch_vectors))
            else:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        metrics.successful_items += result["successful"]
                        metrics.failed_items += result["failed"]
                        metrics.errors.extend(result["errors"])
                    except Exception as e:
                        metrics.failed_items += len(batch_vectors)
                        metrics.errors.append({"error": str(e)})
                        
        # Calculate final metrics
        metrics.total_time = time.time() - start_time
        metrics.average_time_per_item = metrics.total_time / metrics.total_items
        
        # Call callback if provided
        if callback:
            callback(metrics)
            
        return metrics
        
    def _process_batch(
        self,
        vectors: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Process a single batch of vectors.
        
        Args:
            vectors: List of vectors
            metadata: Optional list of metadata dictionaries
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            self.client.insert(vectors, metadata)
            result["successful"] = len(vectors)
        except Exception as e:
            result["failed"] = len(vectors)
            result["errors"].append({
                "error": str(e),
                "batch_size": len(vectors)
            })
            
        return result
        
    def batch_delete(
        self,
        expr: str,
        batch_size: int = 1000,
        callback: Optional[Callable[[BatchMetrics], None]] = None
    ) -> BatchMetrics:
        """
        Delete entities in batches using an expression.
        
        Args:
            expr: Delete expression
            batch_size: Number of items to delete per batch
            callback: Optional callback function for metrics
            
        Returns:
            Batch metrics
        """
        start_time = time.time()
        metrics = BatchMetrics(total_items=0)  # Will be updated after first query
        
        # Get total count
        try:
            total_count = self.client.get_collection_stats()["row_count"]
            metrics.total_items = total_count
        except Exception as e:
            metrics.errors.append({"error": f"Failed to get total count: {str(e)}"})
            return metrics
            
        # Process in batches
        offset = 0
        while offset < total_count:
            try:
                # Delete batch
                self.client.delete(f"{expr} limit {batch_size} offset {offset}")
                metrics.successful_items += batch_size
                offset += batch_size
            except Exception as e:
                metrics.failed_items += batch_size
                metrics.errors.append({
                    "error": str(e),
                    "offset": offset,
                    "batch_size": batch_size
                })
                
        # Calculate final metrics
        metrics.total_time = time.time() - start_time
        metrics.average_time_per_item = metrics.total_time / metrics.total_items
        
        # Call callback if provided
        if callback:
            callback(metrics)
            
        return metrics
        
    def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: int = 10,
        search_params: Optional[Dict[str, Any]] = None,
        callback: Optional[Callable[[BatchMetrics], None]] = None
    ) -> tuple[List[List[Dict[str, Any]]], BatchMetrics]:
        """
        Search for similar vectors in batches.
        
        Args:
            query_vectors: List of query vectors
            top_k: Number of results to return per query
            search_params: Optional search parameters
            callback: Optional callback function for metrics
            
        Returns:
            Tuple of (search results, batch metrics)
        """
        start_time = time.time()
        metrics = BatchMetrics(total_items=len(query_vectors))
        all_results = []
        
        # Create batches
        batches = [
            query_vectors[i:i + self.config.batch_size]
            for i in range(0, len(query_vectors), self.config.batch_size)
        ]
        
        # Process batches
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(
                    self.client.search,
                    batch,
                    top_k,
                    search_params
                )
                futures.append(future)
                
            # Monitor progress
            if self.config.progress_display:
                with tqdm(total=len(query_vectors), desc="Searching vectors") as pbar:
                    for future in as_completed(futures):
                        try:
                            results = future.result()
                            all_results.extend(results)
                            metrics.successful_items += len(results)
                            pbar.update(len(results))
                        except Exception as e:
                            metrics.failed_items += len(batch)
                            metrics.errors.append({"error": str(e)})
                            pbar.update(len(batch))
            else:
                for future in as_completed(futures):
                    try:
                        results = future.result()
                        all_results.extend(results)
                        metrics.successful_items += len(results)
                    except Exception as e:
                        metrics.failed_items += len(batch)
                        metrics.errors.append({"error": str(e)})
                        
        # Calculate final metrics
        metrics.total_time = time.time() - start_time
        metrics.average_time_per_item = metrics.total_time / metrics.total_items
        
        # Call callback if provided
        if callback:
            callback(metrics)
            
        return all_results, metrics 