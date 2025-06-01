"""
Example demonstrating advanced features:
- Streaming with Kafka
- Partition management
- Hybrid search
- Advanced indexing
"""

import numpy as np
import json
import time
from pathlib import Path
import yaml

from ai_prishtina_milvus_client import (
    MilvusConfig,
    AdvancedMilvusClient,
    StreamConfig,
    KafkaStreamProcessor,
    PartitionConfig,
    HybridQueryConfig
)


def generate_sample_data(num_vectors: int = 1000, dim: int = 128) -> tuple:
    """Generate sample vectors and metadata."""
    np.random.seed(42)
    vectors = np.random.rand(num_vectors, dim).tolist()
    metadata = [
        {
            "id": i,
            "category": np.random.choice(["A", "B", "C"]),
            "timestamp": int(time.time()) + i,
            "score": np.random.uniform(0, 1)
        }
        for i in range(num_vectors)
    ]
    return vectors, metadata


def main():
    # Milvus configuration
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="advanced_demo",
        dim=128,
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024
    )
    
    # Stream configuration
    stream_config = StreamConfig(
        bootstrap_servers="localhost:9092",
        group_id="milvus_demo",
        topics=["vector_ingest"],
        batch_size=100,
        num_workers=2
    )
    
    # Initialize advanced client
    client = AdvancedMilvusClient(milvus_config)
    
    try:
        # Create collection
        client.create_collection()
        print("Created collection")
        
        # Create partitions
        partitions = [
            PartitionConfig(
                partition_name="category_A",
                description="Category A vectors",
                tags=["A"]
            ),
            PartitionConfig(
                partition_name="category_B",
                description="Category B vectors",
                tags=["B"]
            ),
            PartitionConfig(
                partition_name="category_C",
                description="Category C vectors",
                tags=["C"]
            )
        ]
        
        for partition in partitions:
            client.create_partition(partition)
            print(f"Created partition: {partition.partition_name}")
            
        # Create index
        client.create_index(
            field_name="vector",
            index_type="IVF_FLAT",
            metric_type="L2",
            params={"nlist": 1024}
        )
        print("Created index")
        
        # Generate and insert sample data
        vectors, metadata = generate_sample_data()
        
        # Insert into appropriate partitions
        for i, (vector, meta) in enumerate(zip(vectors, metadata)):
            partition_name = f"category_{meta['category']}"
            client.insert([vector], [meta], partition_name=partition_name)
            
        print("Inserted sample data")
        
        # Initialize stream processor
        processor = KafkaStreamProcessor(milvus_config, stream_config, client)
        
        # Start streaming in background
        import threading
        stream_thread = threading.Thread(target=processor.start)
        stream_thread.daemon = True
        stream_thread.start()
        
        # Produce some streaming messages
        for i in range(10):
            vectors, metadata = generate_sample_data(num_vectors=10)
            message = {
                "vectors": vectors,
                "metadata": metadata,
                "operation": "insert",
                "collection": "advanced_demo"
            }
            processor.produce_message("vector_ingest", message)
            time.sleep(0.1)
            
        # Wait for messages to be processed
        time.sleep(2)
        
        # Perform hybrid search
        query_config = HybridQueryConfig(
            vector_field="vector",
            scalar_fields=["category", "score"],
            metric_type="L2",
            top_k=5,
            params={"nprobe": 10}
        )
        
        # Search with category filter
        query_vector = vectors[0]  # Use first vector as query
        results = client.hybrid_search(
            [query_vector],
            query_config,
            partition_names=["category_A"],
            category="A",
            score=0.5
        )
        
        print("\nHybrid search results:")
        for i, hit in enumerate(results[0]):
            print(f"Rank {i+1}:")
            print(f"  ID: {hit['id']}")
            print(f"  Category: {hit['category']}")
            print(f"  Score: {hit['score']:.4f}")
            print(f"  Distance: {hit['distance']:.4f}")
            
        # Get partition statistics
        print("\nPartition statistics:")
        for partition in client.list_partitions():
            print(f"\nPartition: {partition['name']}")
            print(f"  Description: {partition['description']}")
            print(f"  Tags: {partition['tags']}")
            print(f"  Entities: {partition['num_entities']}")
            
        # Get index information
        index_info = client.get_index_info("vector")
        print("\nIndex information:")
        print(f"  Type: {index_info['index_type']}")
        print(f"  Metric: {index_info['metric_type']}")
        print(f"  Parameters: {index_info['params']}")
        
    finally:
        # Cleanup
        client.drop_collection()
        print("\nCleaned up collection")


if __name__ == "__main__":
    main() 