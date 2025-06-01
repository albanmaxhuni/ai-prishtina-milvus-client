"""
Example: Distributed processing and caching with Milvus.
- Processes large datasets in parallel
- Uses Redis for distributed caching
- Demonstrates performance improvements
"""

import os
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel

from ai_prishtina_milvus_client import (
    MilvusConfig,
    DistributedMilvusClient,
    CacheConfig,
    DistributedConfig
)


class TextProcessor:
    """Process text data for vector generation."""
    
    def __init__(self):
        # Initialize text embedding model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.numpy().tolist()


def generate_sample_data(num_items: int) -> tuple[List[str], List[Dict[str, Any]]]:
    """Generate sample text data and metadata."""
    texts = [
        f"This is sample text {i} for testing distributed processing and caching."
        for i in range(num_items)
    ]
    
    metadata = [
        {
            "id": i,
            "category": f"category_{i % 5}",
            "score": np.random.uniform(0, 1),
            "timestamp": time.time()
        }
        for i in range(num_items)
    ]
    
    return texts, metadata


def main():
    # Initialize text processor
    processor = TextProcessor()
    
    # Milvus configuration
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="distributed_demo",
        dim=384,  # MiniLM-L6-v2 dimension
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024
    )
    
    # Cache configuration
    cache_config = CacheConfig(
        enabled=True,
        redis_url="redis://localhost:6379",
        ttl=3600,
        max_size=1000
    )
    
    # Distributed processing configuration
    distributed_config = DistributedConfig(
        enabled=True,
        num_workers=os.cpu_count() or 4,
        chunk_size=100,
        use_processes=True,
        timeout=300
    )
    
    # Initialize distributed client
    client = DistributedMilvusClient(
        milvus_config,
        cache_config=cache_config,
        distributed_config=distributed_config
    )
    
    try:
        # Create collection
        client.create_collection()
        print("Created collection")
        
        # Generate sample data
        num_items = 10000
        print(f"\nGenerating {num_items} sample items...")
        texts, metadata = generate_sample_data(num_items)
        
        # Generate embeddings
        print("Generating embeddings...")
        start_time = time.time()
        vectors = processor.generate_embeddings(texts)
        embedding_time = time.time() - start_time
        print(f"Generated {len(vectors)} embeddings in {embedding_time:.2f} seconds")
        
        # Insert vectors with distributed processing
        print("\nInserting vectors with distributed processing...")
        start_time = time.time()
        client.insert_vectors(vectors, metadata)
        insert_time = time.time() - start_time
        print(f"Inserted {len(vectors)} vectors in {insert_time:.2f} seconds")
        
        # Perform search with caching
        print("\nPerforming search with caching...")
        query_text = "This is a sample query for testing"
        query_vector = processor.generate_embeddings([query_text])[0]
        
        # First search (cache miss)
        start_time = time.time()
        results1 = client.search_vectors([query_vector], top_k=5)
        search_time1 = time.time() - start_time
        print(f"First search (cache miss) completed in {search_time1:.2f} seconds")
        
        # Second search (cache hit)
        start_time = time.time()
        results2 = client.search_vectors([query_vector], top_k=5)
        search_time2 = time.time() - start_time
        print(f"Second search (cache hit) completed in {search_time2:.2f} seconds")
        
        # Print search results
        print("\nSearch results:")
        for i, hit in enumerate(results1[0]):
            print(f"Rank {i+1}:")
            print(f"  ID: {hit['id']}")
            print(f"  Category: {hit['category']}")
            print(f"  Score: {hit['score']:.4f}")
            print(f"  Distance: {hit['distance']:.4f}")
        
        # Get cache statistics
        print("\nCache statistics:")
        stats = client.get_cache_stats()
        print(f"LRU Cache:")
        print(f"  Size: {stats['lru_cache_size']}")
        print(f"  Hits: {stats['lru_cache_hits']}")
        print(f"  Misses: {stats['lru_cache_misses']}")
        
        if "redis_used_memory" in stats:
            print(f"\nRedis Cache:")
            print(f"  Used Memory: {stats['redis_used_memory'] / 1024 / 1024:.2f} MB")
            print(f"  Connected Clients: {stats['redis_connected_clients']}")
            print(f"  Commands Processed: {stats['redis_commands_processed']}")
        
        # Clear cache
        print("\nClearing cache...")
        client.clear_cache()
        print("Cache cleared")
        
    finally:
        # Cleanup
        client.drop_collection()
        print("\nCleaned up collection")


if __name__ == "__main__":
    main() 