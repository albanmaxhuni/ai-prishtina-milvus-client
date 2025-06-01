"""
Example demonstrating different data formats.
"""

import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def create_sample_data():
    """Create sample data files in different formats."""
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
    data = [
        {"vector": v, **m}
        for v, m in zip(vectors, metadata)
    ]
    with open("sample_data.json", "w") as f:
        json.dump(data, f, indent=2)
        
    # Create NumPy file
    np.savez(
        "sample_data.npz",
        vector=np.array(vectors),
        category=np.array([m["category"] for m in metadata]),
        score=np.array([m["score"] for m in metadata]),
        tags=np.array([m["tags"] for m in metadata])
    )
    
    # Create HDF5 file
    with h5py.File("sample_data.h5", "w") as f:
        f.create_dataset("vector", data=vectors)
        f.create_dataset("category", data=[m["category"] for m in metadata])
        f.create_dataset("score", data=[m["score"] for m in metadata])
        f.create_dataset("tags", data=[m["tags"] for m in metadata])
        
    # Create Parquet file
    df = pd.DataFrame({
        "vector": vectors,
        "category": [m["category"] for m in metadata],
        "score": [m["score"] for m in metadata],
        "tags": [m["tags"] for m in metadata]
    })
    df.to_parquet("sample_data.parquet")
    
    # Create Pickle file
    with open("sample_data.pkl", "wb") as f:
        pickle.dump(data, f)
        
    # Create YAML file
    with open("sample_data.yaml", "w") as f:
        yaml.dump(data, f)


def create_data_source_configs():
    """Create data source configuration files."""
    # Common configuration
    base_config = {
        "vector_field": "vector",
        "metadata_fields": ["category", "score", "tags"],
        "batch_size": 1000
    }
    
    # CSV configuration
    csv_config = {
        **base_config,
        "type": "csv",
        "path": "sample_data.csv"
    }
    with open("csv_config.yaml", "w") as f:
        yaml.dump(csv_config, f)
        
    # JSON configuration
    json_config = {
        **base_config,
        "type": "json",
        "path": "sample_data.json"
    }
    with open("json_config.yaml", "w") as f:
        yaml.dump(json_config, f)
        
    # NumPy configuration
    numpy_config = {
        **base_config,
        "type": "numpy",
        "path": "sample_data.npz"
    }
    with open("numpy_config.yaml", "w") as f:
        yaml.dump(numpy_config, f)
        
    # HDF5 configuration
    hdf5_config = {
        **base_config,
        "type": "hdf5",
        "path": "sample_data.h5"
    }
    with open("hdf5_config.yaml", "w") as f:
        yaml.dump(hdf5_config, f)
        
    # Parquet configuration
    parquet_config = {
        **base_config,
        "type": "parquet",
        "path": "sample_data.parquet"
    }
    with open("parquet_config.yaml", "w") as f:
        yaml.dump(parquet_config, f)
        
    # Pickle configuration
    pickle_config = {
        **base_config,
        "type": "pickle",
        "path": "sample_data.pkl"
    }
    with open("pickle_config.yaml", "w") as f:
        yaml.dump(pickle_config, f)
        
    # YAML configuration
    yaml_config = {
        **base_config,
        "type": "yaml",
        "path": "sample_data.yaml"
    }
    with open("yaml_config.yaml", "w") as f:
        yaml.dump(yaml_config, f)


def main():
    # Create sample data and configurations
    print("Creating sample data and configurations...")
    create_sample_data()
    create_data_source_configs()
    
    # Create Milvus configuration
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="data_formats_example",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    
    config_path = Path("config.yaml")
    config.to_yaml(config_path)
    
    # Process data from different formats
    with MilvusClient(config_path) as client:
        # Create collection
        client.create_collection()
        
        # Process each format
        formats = [
            ("CSV", "csv_config.yaml"),
            ("JSON", "json_config.yaml"),
            ("NumPy", "numpy_config.yaml"),
            ("HDF5", "hdf5_config.yaml"),
            ("Parquet", "parquet_config.yaml"),
            ("Pickle", "pickle_config.yaml"),
            ("YAML", "yaml_config.yaml")
        ]
        
        for format_name, config_file in formats:
            print(f"\nProcessing {format_name} format...")
            
            # Insert vectors
            client.insert_from_source(config_file)
            
            # Get collection statistics
            stats = client.get_collection_stats()
            print(f"Collection statistics: {stats}")
            
            # Search vectors
            query_vector = np.random.rand(128).tolist()
            results = client.search([query_vector], top_k=5)
            
            print(f"\nSearch results for {format_name}:")
            for i, result in enumerate(results[0]):
                print(f"  Result {i + 1}:")
                print(f"    ID: {result['id']}")
                print(f"    Distance: {result['distance']:.4f}")
                
        # Drop collection
        client.drop_collection()
        
    # Clean up
    print("\nCleaning up...")
    for file in Path(".").glob("sample_data.*"):
        file.unlink()
    for file in Path(".").glob("*_config.yaml"):
        file.unlink()
    config_path.unlink()


if __name__ == "__main__":
    main() 