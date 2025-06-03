"""
Example demonstrating collection management features.
"""

import numpy as np
from pathlib import Path

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def main():
    # Create a configuration file
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="management_example",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    
    config_path = Path("config.yaml")
    config.to_yaml(config_path)
    
    # Using context manager for automatic cleanup
    with MilvusClient(config_path) as client:
        # List existing collections
        print("Existing collections:")
        collections = client.list_collections()
        print(f"  {collections}")
        
        # Create collection
        print("\nCreating collection...")
        client.create_collection()
        
        # Insert some vectors
        print("Inserting vectors...")
        vectors = np.random.rand(100, 128).tolist()
        client.insert(vectors)
        
        # Get collection statistics
        print("\nCollection statistics:")
        stats = client.get_collection_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        # Search vectors
        print("\nSearching vectors...")
        query_vector = vectors[0]
        results = client.search([query_vector], top_k=5)
        
        print("\nSearch results:")
        for i, result in enumerate(results[0]):
            print(f"  Result {i + 1}:")
            print(f"    ID: {result['id']}")
            print(f"    Distance: {result['distance']:.4f}")
            
        # Drop collection
        print("\nDropping collection...")
        client.drop_collection()
        
        # Verify collection is dropped
        collections = client.list_collections()
        print(f"\nCollections after dropping: {collections}")
        
    # Clean up config file
    config_path.unlink()


if __name__ == "__main__":
    main() 