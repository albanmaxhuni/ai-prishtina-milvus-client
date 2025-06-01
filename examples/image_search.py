"""
Example: Image search with hybrid filtering.
- Ingests image embeddings with metadata
- Performs hybrid search by image similarity and metadata
- Uses partitions for different image categories
"""

import numpy as np
from pathlib import Path
import json
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from ai_prishtina_milvus_client import (
    MilvusConfig,
    AdvancedMilvusClient,
    PartitionConfig,
    HybridQueryConfig
)


class ImageEmbedder:
    """Image embedding model using ResNet50."""
    
    def __init__(self):
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
    def embed_image(self, image_path: str) -> List[float]:
        """Generate embedding for an image."""
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            embedding = self.model(image_tensor)
        return embedding.squeeze().numpy().tolist()


def main():
    # Initialize image embedder
    embedder = ImageEmbedder()
    
    # Milvus configuration
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="image_search",
        dim=2048,  # ResNet50 feature dimension
        index_type="IVF_FLAT",
        metric_type="L2",
        nlist=1024
    )
    
    # Initialize advanced client
    client = AdvancedMilvusClient(milvus_config)
    
    try:
        # Create collection
        client.create_collection()
        print("Created collection")
        
        # Create partitions for different image categories
        categories = ["nature", "people", "animals", "objects"]
        for category in categories:
            partition = PartitionConfig(
                partition_name=f"category_{category}",
                description=f"{category.capitalize()} images",
                tags=[category]
            )
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
        
        # Example: Ingest images from a directory
        image_dir = Path("sample_images")
        if not image_dir.exists():
            print("Please create a 'sample_images' directory with some images")
            return
            
        # Process and insert images
        for image_path in image_dir.glob("*.jpg"):
            # Generate embedding
            vector = embedder.embed_image(str(image_path))
            
            # Create metadata
            metadata = {
                "filename": image_path.name,
                "category": "nature" if "nature" in image_path.name.lower() else "objects",
                "size": image_path.stat().st_size,
                "timestamp": image_path.stat().st_mtime
            }
            
            # Insert into appropriate partition
            partition_name = f"category_{metadata['category']}"
            client.insert([vector], [metadata], partition_name=partition_name)
            print(f"Inserted {image_path.name}")
            
        # Perform hybrid search
        query_config = HybridQueryConfig(
            vector_field="vector",
            scalar_fields=["category", "size"],
            metric_type="L2",
            top_k=5,
            params={"nprobe": 10}
        )
        
        # Example: Search for similar images in nature category
        query_image = next(image_dir.glob("*.jpg"))
        query_vector = embedder.embed_image(str(query_image))
        
        results = client.hybrid_search(
            [query_vector],
            query_config,
            partition_names=["category_nature"],
            category="nature"
        )
        
        print("\nSearch results:")
        for i, hit in enumerate(results[0]):
            print(f"Rank {i+1}:")
            print(f"  Image: {hit['filename']}")
            print(f"  Category: {hit['category']}")
            print(f"  Size: {hit['size']} bytes")
            print(f"  Distance: {hit['distance']:.4f}")
            
        # Get partition statistics
        print("\nPartition statistics:")
        for partition in client.list_partitions():
            print(f"\nPartition: {partition['name']}")
            print(f"  Description: {partition['description']}")
            print(f"  Tags: {partition['tags']}")
            print(f"  Entities: {partition['num_entities']}")
            
    finally:
        # Cleanup
        client.drop_collection()
        print("\nCleaned up collection")


if __name__ == "__main__":
    main() 