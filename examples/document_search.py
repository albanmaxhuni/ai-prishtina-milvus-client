"""
Example: Real-world document search pipeline using MilvusClient.
- Ingests a set of text documents (e.g., news articles)
- Generates embeddings (mocked for demo)
- Stores in Milvus
- Performs semantic search
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient

# Mock embedding function (replace with real API in production)
def embed_texts(texts, dim=384):
    np.random.seed(42)
    return np.random.rand(len(texts), dim).tolist()


def main():
    # Example documents (news headlines)
    docs = [
        "AI revolutionizes healthcare with faster diagnostics",
        "Stock markets rally as tech shares surge",
        "Climate change impacts global food supply",
        "New species discovered in the Amazon rainforest",
        "Breakthrough in quantum computing announced",
        "Sports teams prepare for the summer Olympics",
        "Scientists develop new vaccine for rare disease",
        "Electric vehicles gain popularity worldwide",
        "Major cybersecurity breach affects millions",
        "Space mission sets sights on Mars landing",
    ]
    sources = [
        "health",
        "finance",
        "environment",
        "science",
        "technology",
        "sports",
        "health",
        "technology",
        "security",
        "space",
    ]
    # Generate embeddings
    vectors = embed_texts(docs, dim=384)
    metadata = [{"headline": doc, "category": cat} for doc, cat in zip(docs, sources)]
    # Save as CSV
    df = pd.DataFrame({
        "vector": [str(v) for v in vectors],
        "headline": [m["headline"] for m in metadata],
        "category": [m["category"] for m in metadata],
    })
    csv_path = Path("news_articles.csv")
    df.to_csv(csv_path, index=False)
    # Data source config
    config = {
        "vector_field": "vector",
        "metadata_fields": ["headline", "category"],
        "batch_size": 1000,
        "type": "csv",
        "path": str(csv_path),
    }
    config_path = Path("news_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    # Milvus config
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="news_search",
        dim=384,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        print("Ingested news articles into Milvus.")
        # Semantic search
        query = "AI in medicine"
        query_vector = embed_texts([query], dim=384)[0]
        results = client.search([query_vector], top_k=3)
        print(f"\nTop 3 results for query: '{query}'")
        for i, r in enumerate(results[0]):
            print(f"  Rank {i+1}: {r['headline']} (Category: {r['category']}, Distance: {r['distance']:.4f})")
        # Hybrid search: restrict to technology
        results = client.search([query_vector], top_k=3, filter="category == 'technology'")
        print(f"\nTop results for query in 'technology':")
        for i, r in enumerate(results[0]):
            print(f"  Rank {i+1}: {r['headline']} (Distance: {r['distance']:.4f})")
        client.drop_collection()
    # Clean up
    csv_path.unlink()
    config_path.unlink()

if __name__ == "__main__":
    main() 