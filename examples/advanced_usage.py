"""
Advanced usage example for the Milvus client with metadata handling.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def generate_sample_data(num_vectors: int, dim: int) -> tuple[List[List[float]], List[Dict[str, Any]]]:
    """Generate sample vectors and metadata."""
    vectors = np.random.rand(num_vectors, dim).tolist()
    metadata = [
        {
            "category": np.random.choice(["A", "B", "C"]),
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3", "tag4"], size=2).tolist(),
        }
        for _ in range(num_vectors)
    ]
    return vectors, metadata


def main():
    # Create a configuration file
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="advanced_collection",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    
    config_path = Path("config.yaml")
    config.to_yaml(config_path)
    
    # Initialize client
    client = MilvusClient(config_path)
    
    try:
        # Create collection
        print("Creating collection...")
        client.create_collection()
        
        # Generate sample data
        print("Generating sample data...")
        vectors, metadata = generate_sample_data(1000, 128)
        
        # Insert vectors with metadata
        print("Inserting vectors with metadata...")
        client.insert(vectors, metadata)
        
        # Search for similar vectors
        print("\nSearching for similar vectors...")
        query_vector = vectors[0]
        results = client.search([query_vector], top_k=5)
        
        # Print results with metadata
        print("\nSearch results:")
        for i, result in enumerate(results[0]):
            print(f"Result {i + 1}:")
            print(f"  ID: {result['id']}")
            print(f"  Distance: {result['distance']:.4f}")
            if "metadata" in result:
                print("  Metadata:")
                for key, value in result["metadata"].items():
                    print(f"    {key}: {value}")
                    
        # Delete vectors by category
        print("\nDeleting vectors with category 'A'...")
        client.delete("category == 'A'")
        
        # Verify deletion
        results = client.search([query_vector], top_k=10)
        print(f"\nRemaining results after deletion: {len(results[0])}")
        
    finally:
        # Clean up
        client.close()
        config_path.unlink()


if __name__ == "__main__":
    main() 