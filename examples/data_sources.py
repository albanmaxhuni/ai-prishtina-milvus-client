"""
Example demonstrating different data sources.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def create_sample_data():
    """Create sample data files for different formats."""
    # Create sample vectors and metadata
    vectors = np.random.rand(100, 128).tolist()
    metadata = [
        {
            "category": np.random.choice(["A", "B", "C"]),
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
        }
        for _ in range(100)
    ]
    
    # Create CSV file
    df = pd.DataFrame({
        "vector": [str(v) for v in vectors],
        "category": [m["category"] for m in metadata],
        "score": [m["score"] for m in metadata],
        "tags": [str(m["tags"]) for m in metadata],
    })
    df.to_csv("sample_data.csv", index=False)
    
    # Create JSON file
    json_data = [
        {"vector": v, **m}
        for v, m in zip(vectors, metadata)
    ]
    with open("sample_data.json", "w") as f:
        json.dump(json_data, f)
        
    # Create NumPy file
    np.savez(
        "sample_data.npz",
        vector=np.array(vectors),
        category=np.array([m["category"] for m in metadata]),
        score=np.array([m["score"] for m in metadata]),
        tags=np.array([m["tags"] for m in metadata]),
    )
    
    # Create data source configs
    configs = {
        "csv": {
            "type": "csv",
            "path": "sample_data.csv",
            "vector_field": "vector",
            "metadata_fields": ["category", "score", "tags"],
        },
        "json": {
            "type": "json",
            "path": "sample_data.json",
            "vector_field": "vector",
            "metadata_fields": ["category", "score", "tags"],
        },
        "numpy": {
            "type": "numpy",
            "path": "sample_data.npz",
            "vector_field": "vector",
            "metadata_fields": ["category", "score", "tags"],
        },
    }
    
    for source_type, config in configs.items():
        with open(f"data_source_{source_type}.json", "w") as f:
            json.dump(config, f, indent=2)


def main():
    # Create sample data
    print("Creating sample data...")
    create_sample_data()
    
    # Create Milvus configuration
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="data_sources_example",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    
    config_path = Path("config.yaml")
    config.to_yaml(config_path)
    
    # Process each data source
    with MilvusClient(config_path) as client:
        for source_type in ["csv", "json", "numpy"]:
            print(f"\nProcessing {source_type.upper()} data source...")
            
            # Create collection
            client.create_collection()
            
            # Insert data from source
            print(f"Inserting data from {source_type}...")
            client.insert_from_source(f"data_source_{source_type}.json")
            
            # Get collection statistics
            stats = client.get_collection_stats()
            print(f"Collection statistics: {stats}")
            
            # Search vectors
            query_vector = np.random.rand(128).tolist()
            results = client.search([query_vector], top_k=5)
            
            print(f"\nSearch results from {source_type}:")
            for i, result in enumerate(results[0]):
                print(f"  Result {i + 1}:")
                print(f"    ID: {result['id']}")
                print(f"    Distance: {result['distance']:.4f}")
                
            # Drop collection for next iteration
            client.drop_collection()
            
    # Clean up
    print("\nCleaning up...")
    for file in Path(".").glob("sample_data.*"):
        file.unlink()
    for file in Path(".").glob("data_source_*.json"):
        file.unlink()
    config_path.unlink()


if __name__ == "__main__":
    main() 