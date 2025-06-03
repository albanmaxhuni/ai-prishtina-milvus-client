"""
Example: Multi-modal search combining text and image embeddings.
- Processes images and text together
- Generates embeddings for both modalities
- Stores in Milvus with metadata
- Performs hybrid search across modalities
"""

import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    AutoModel,
    CLIPProcessor,
    CLIPModel
)
import json

from ai_prishtina_milvus_client import (
    MilvusConfig,
    AdvancedMilvusClient,
    PartitionConfig,
    HybridQueryConfig
)


class MultiModalProcessor:
    """Process both text and images for multi-modal search."""
    
    def __init__(self):
        # Initialize text embedding model
        self.text_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_model.eval()
        
        # Initialize CLIP model for image-text embeddings
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval()
        
    def generate_text_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        inputs = self.text_tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.squeeze().numpy().tolist()
    
    def generate_image_embedding(self, image_path: str) -> List[float]:
        """Generate embedding for image using CLIP."""
        image = Image.open(image_path).convert('RGB')
        inputs = self.clip_processor(
            images=image,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.clip_model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)
            
        return embeddings.squeeze().numpy().tolist()
    
    def generate_text_image_embedding(self, text: str, image_path: str) -> Dict[str, List[float]]:
        """Generate embeddings for both text and image."""
        return {
            "text_embedding": self.generate_text_embedding(text),
            "image_embedding": self.generate_image_embedding(image_path)
        }


def main():
    # Initialize multi-modal processor
    processor = MultiModalProcessor()
    
    # Milvus configuration
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="multimodal_search",
        dim=512,  # Combined dimension for text and image
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
        
        # Create partitions for different content types
        content_types = ["products", "articles", "social_media", "other"]
        for content_type in content_types:
            partition = PartitionConfig(
                partition_name=f"type_{content_type}",
                description=f"{content_type.capitalize()} content",
                tags=[content_type]
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
        
        # Example: Process content from a directory
        content_dir = Path("sample_content")
        if not content_dir.exists():
            print("Please create a 'sample_content' directory with images and text files")
            return
        
        # Process and insert content
        for item_path in content_dir.glob("*"):
            if item_path.suffix.lower() in ['.jpg', '.png', '.jpeg']:
                print(f"\nProcessing {item_path.name}...")
                
                # Get corresponding text file
                text_path = item_path.with_suffix('.txt')
                if not text_path.exists():
                    print(f"No text file found for {item_path.name}")
                    continue
                
                # Read text content
                with open(text_path, 'r') as f:
                    text_content = f.read()
                
                # Generate embeddings
                embeddings = processor.generate_text_image_embedding(
                    text_content,
                    str(item_path)
                )
                
                # Create metadata
                metadata = {
                    "filename": item_path.name,
                    "content_type": "products" if "product" in item_path.name.lower() else "other",
                    "text_length": len(text_content),
                    "image_size": Image.open(item_path).size
                }
                
                # Combine embeddings
                combined_vector = np.concatenate([
                    embeddings["text_embedding"],
                    embeddings["image_embedding"]
                ]).tolist()
                
                # Insert into appropriate partition
                partition_name = f"type_{metadata['content_type']}"
                client.insert([combined_vector], [metadata], partition_name=partition_name)
                print(f"Inserted {item_path.name}")
        
        # Perform hybrid search
        query_config = HybridQueryConfig(
            vector_field="vector",
            scalar_fields=["content_type", "text_length"],
            metric_type="L2",
            top_k=5,
            params={"nprobe": 10}
        )
        
        # Example: Search for similar content in products
        query_text = "Example search query about products"
        query_image = next(content_dir.glob("*.jpg"))
        
        # Generate query embeddings
        query_embeddings = processor.generate_text_image_embedding(
            query_text,
            str(query_image)
        )
        
        # Combine query embeddings
        query_vector = np.concatenate([
            query_embeddings["text_embedding"],
            query_embeddings["image_embedding"]
        ]).tolist()
        
        results = client.hybrid_search(
            [query_vector],
            query_config,
            partition_names=["type_products"],
            content_type="products"
        )
        
        print("\nSearch results:")
        for i, hit in enumerate(results[0]):
            print(f"Rank {i+1}:")
            print(f"  Content: {hit['filename']}")
            print(f"  Type: {hit['content_type']}")
            print(f"  Text Length: {hit['text_length']}")
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