"""
Example demonstrating cloud storage and API integrations.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
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
    
    # Create data source config
    config = {
        "type": "csv",
        "path": "sample_data.csv",
        "vector_field": "vector",
        "metadata_fields": ["category", "score", "tags"],
    }
    with open("data_source.json", "w") as f:
        json.dump(config, f, indent=2)


def create_cloud_config():
    """Create cloud storage configuration files."""
    # AWS S3 config
    s3_config = {
        "provider": "aws",
        "bucket": "my-bucket",
        "prefix": "vectors/",
        "region": "us-west-2",
        "credentials": {
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    }
    with open("s3_config.json", "w") as f:
        json.dump(s3_config, f, indent=2)
        
    # GCP config
    gcp_config = {
        "provider": "gcp",
        "bucket": "my-bucket",
        "prefix": "vectors/",
        "credentials": {
            "type": "service_account",
            "project_id": os.getenv("GCP_PROJECT_ID"),
            "private_key_id": os.getenv("GCP_PRIVATE_KEY_ID"),
            "private_key": os.getenv("GCP_PRIVATE_KEY"),
            "client_email": os.getenv("GCP_CLIENT_EMAIL"),
            "client_id": os.getenv("GCP_CLIENT_ID"),
        }
    }
    with open("gcp_config.json", "w") as f:
        json.dump(gcp_config, f, indent=2)


def create_api_config():
    """Create API configuration files."""
    # OpenAI config
    openai_config = {
        "service": "openai",
        "base_url": "https://api.openai.com",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "timeout": 30
    }
    with open("openai_config.json", "w") as f:
        json.dump(openai_config, f, indent=2)
        
    # HuggingFace config
    hf_config = {
        "service": "huggingface",
        "base_url": "https://api-inference.huggingface.co",
        "api_key": os.getenv("HF_API_KEY"),
        "timeout": 30
    }
    with open("hf_config.json", "w") as f:
        json.dump(hf_config, f, indent=2)


def main():
    # Create sample data and configurations
    print("Creating sample data and configurations...")
    create_sample_data()
    create_cloud_config()
    create_api_config()
    
    # Create Milvus configuration
    config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="cloud_api_example",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    
    config_path = Path("config.yaml")
    config.to_yaml(config_path)
    
    # Process data from different sources
    with MilvusClient(config_path) as client:
        # Create collection
        client.create_collection()
        
        # 1. Insert from local data source
        print("\nInserting from local data source...")
        client.insert_from_source("data_source.json")
        
        # 2. Insert from cloud storage (if credentials are available)
        if os.getenv("AWS_ACCESS_KEY_ID"):
            print("\nInserting from AWS S3...")
            client.insert_from_cloud("s3_config.json", "vectors/sample_data.csv")
            
        if os.getenv("GCP_PROJECT_ID"):
            print("\nInserting from GCP...")
            client.insert_from_cloud("gcp_config.json", "vectors/sample_data.csv")
            
        # 3. Insert from APIs (if credentials are available)
        if os.getenv("OPENAI_API_KEY"):
            print("\nInserting from OpenAI...")
            client.insert_from_api(
                "openai_config.json",
                "This is a sample text for embedding",
                model="text-embedding-ada-002"
            )
            
        if os.getenv("HF_API_KEY"):
            print("\nInserting from HuggingFace...")
            client.insert_from_api(
                "hf_config.json",
                "This is a sample text for embedding",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
        # Get collection statistics
        stats = client.get_collection_stats()
        print(f"\nCollection statistics: {stats}")
        
        # Search vectors
        query_vector = np.random.rand(128).tolist()
        results = client.search([query_vector], top_k=5)
        
        print("\nSearch results:")
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
    for file in Path(".").glob("*_config.json"):
        file.unlink()
    config_path.unlink()


if __name__ == "__main__":
    main() 