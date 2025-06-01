"""
Advanced integration test: end-to-end file ingestion, search, and validation using MilvusClient.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ai_prishtina_milvus_client import MilvusConfig, MilvusClient


def create_temp_csv(vectors, metadata):
    df = pd.DataFrame({
        "vector": [str(v) for v in vectors],
        "category": [m["category"] for m in metadata],
        "score": [m["score"] for m in metadata],
        "tags": [str(m["tags"]) for m in metadata],
    })
    temp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(temp.name, index=False)
    return temp.name


def create_temp_config(csv_path, collection_name="integration_test_collection"):
    config = {
        "vector_field": "vector",
        "metadata_fields": ["category", "score", "tags"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    temp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
    yaml.dump(config, temp)
    return temp.name


def create_milvus_config(collection_name="integration_test_collection"):
    return MilvusConfig(
        host="localhost",
        port=19530,
        collection_name=collection_name,
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )


def test_end_to_end_file_ingestion_and_search():
    # Generate test data
    vectors = np.random.rand(50, 128).tolist()
    metadata = [
        {
            "category": np.random.choice(["A", "B", "C"]),
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
        }
        for _ in range(50)
    ]
    csv_path = create_temp_csv(vectors, metadata)
    config_path = create_temp_config(csv_path)
    milvus_config = create_milvus_config()

    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 50
        query_vector = vectors[0]
        results = client.search([query_vector], top_k=5)
        assert len(results) == 1
        assert len(results[0]) == 5
        for r in results[0]:
            assert "id" in r and "distance" in r
        client.drop_collection()

    # Clean up
    os.unlink(csv_path)
    os.unlink(config_path)


def test_hybrid_search_with_metadata_filter():
    # Generate test data
    vectors = np.random.rand(50, 128).tolist()
    categories = [np.random.choice(["A", "B", "C"]) for _ in range(50)]
    metadata = [
        {
            "category": cat,
            "score": float(np.random.randint(0, 100)),
            "tags": np.random.choice(["tag1", "tag2", "tag3"], size=2).tolist(),
        }
        for cat in categories
    ]
    csv_path = create_temp_csv(vectors, metadata)
    config_path = create_temp_config(csv_path, collection_name="hybrid_test_collection")
    milvus_config = create_milvus_config(collection_name="hybrid_test_collection")

    # Ingest and hybrid search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 50
        query_vector = vectors[0]
        target_category = metadata[0]["category"]
        # Hybrid search: vector + metadata filter
        results = client.search(
            [query_vector],
            top_k=5,
            filter=f"category == '{target_category}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["category"] == target_category
        client.drop_collection()

    # Clean up
    os.unlink(csv_path)
    os.unlink(config_path)


def test_multimodal_image_text_ingestion_and_search():
    # Generate test data
    image_vectors = np.random.rand(30, 256).tolist()  # e.g., image embeddings
    captions = [f"A photo of a {obj}" for obj in np.random.choice(["cat", "dog", "car", "tree"], size=30)]
    metadata = [
        {
            "caption": cap,
            "source": np.random.choice(["web", "user", "dataset"]),
        }
        for cap in captions
    ]
    # Save as CSV
    df = pd.DataFrame({
        "vector": [str(v) for v in image_vectors],
        "caption": [m["caption"] for m in metadata],
        "source": [m["source"] for m in metadata],
    })
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df.to_csv(csv_path, index=False)
    # Config
    config = {
        "vector_field": "vector",
        "metadata_fields": ["caption", "source"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    config_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="multimodal_image_text",
        dim=256,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 30
        query_vector = image_vectors[0]
        target_caption = metadata[0]["caption"]
        # Search by vector and filter by caption
        results = client.search(
            [query_vector],
            top_k=3,
            filter=f"caption == '{target_caption}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["caption"] == target_caption
        client.drop_collection()
    os.unlink(csv_path)
    os.unlink(config_path)


def test_multimodal_audio_text_ingestion_and_search():
    # Generate test data
    audio_vectors = np.random.rand(20, 128).tolist()  # e.g., audio embeddings
    transcripts = [
        f"This is a recording of a {obj}."
        for obj in np.random.choice(["meeting", "lecture", "podcast", "interview"], size=20)
    ]
    metadata = [
        {
            "transcript": t,
            "duration": float(np.random.randint(30, 300)),  # seconds
        }
        for t in transcripts
    ]
    # Save as CSV
    df = pd.DataFrame({
        "vector": [str(v) for v in audio_vectors],
        "transcript": [m["transcript"] for m in metadata],
        "duration": [m["duration"] for m in metadata],
    })
    csv_path = tempfile.NamedTemporaryFile(suffix=".csv", delete=False).name
    df.to_csv(csv_path, index=False)
    # Config
    config = {
        "vector_field": "vector",
        "metadata_fields": ["transcript", "duration"],
        "batch_size": 1000,
        "type": "csv",
        "path": csv_path,
    }
    config_path = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False).name
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="multimodal_audio_text",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024,
    )
    # Ingest and search
    with MilvusClient(milvus_config) as client:
        client.create_collection()
        client.insert_from_source(config_path)
        stats = client.get_collection_stats()
        assert stats["row_count"] >= 20
        query_vector = audio_vectors[0]
        target_transcript = metadata[0]["transcript"]
        # Search by vector and filter by transcript
        results = client.search(
            [query_vector],
            top_k=2,
            filter=f"transcript == '{target_transcript}'"
        )
        assert len(results) == 1
        for r in results[0]:
            assert r["transcript"] == target_transcript
        client.drop_collection()
    os.unlink(csv_path)
    os.unlink(config_path) 