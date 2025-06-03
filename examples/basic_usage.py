"""
Basic usage example for the Milvus client.
"""

import numpy as np
from pathlib import Path

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def main():
    # Create a configuration file
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="example_collection",
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
        
        # Generate some random vectors
        print("Generating test vectors...")
        vectors = np.random.rand(1000, 128).tolist()
        
        # Insert vectors
        print("Inserting vectors...")
        client.insert(vectors)
        
        # Search for similar vectors
        print("Searching for similar vectors...")
        query_vector = vectors[0]
        results = client.search([query_vector], top_k=5)
        
        # Print results
        print("\nSearch results:")
        for i, result in enumerate(results[0]):
            print(f"Result {i + 1}:")
            print(f"  ID: {result['id']}")
            print(f"  Distance: {result['distance']:.4f}")
            
    finally:
        # Clean up
        client.close()
        config_path.unlink()


if __name__ == "__main__":
    main() 