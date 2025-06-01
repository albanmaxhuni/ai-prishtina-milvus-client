"""
Example: PDF document processing with OCR and vector search.
- Extracts text from PDFs using OCR
- Generates embeddings for text content
- Stores in Milvus with metadata
- Performs hybrid search by content and metadata
"""

import os
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModel
import json

from ai_prishtina_milvus_client import (
    MilvusConfig,
    AdvancedMilvusClient,
    PartitionConfig,
    HybridQueryConfig
)


class PDFProcessor:
    """Process PDF documents with OCR and text embedding."""
    
    def __init__(self):
        # Initialize text embedding model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model.eval()
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF using OCR."""
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        # Process each page
        pages = []
        for i, image in enumerate(images):
            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Store page information
            pages.append({
                "page_number": i + 1,
                "text": text,
                "image_size": image.size
            })
            
        return pages
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using transformer model."""
        # Tokenize and prepare input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Generate embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.squeeze().numpy().tolist()


def main():
    # Initialize PDF processor
    processor = PDFProcessor()
    
    # Milvus configuration
    milvus_config = MilvusConfig(
        host="localhost",
        port=19530,
        collection_name="pdf_search",
        dim=384,  # MiniLM-L6-v2 dimension
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
        
        # Create partitions for different document types
        doc_types = ["reports", "invoices", "contracts", "other"]
        for doc_type in doc_types:
            partition = PartitionConfig(
                partition_name=f"type_{doc_type}",
                description=f"{doc_type.capitalize()} documents",
                tags=[doc_type]
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
        
        # Example: Process PDFs from a directory
        pdf_dir = Path("sample_pdfs")
        if not pdf_dir.exists():
            print("Please create a 'sample_pdfs' directory with some PDF files")
            return
        
        # Process and insert PDFs
        for pdf_path in pdf_dir.glob("*.pdf"):
            print(f"\nProcessing {pdf_path.name}...")
            
            # Extract text from PDF
            pages = processor.extract_text_from_pdf(str(pdf_path))
            
            # Process each page
            for page in pages:
                # Generate embedding for page text
                vector = processor.generate_embedding(page["text"])
                
                # Create metadata
                metadata = {
                    "filename": pdf_path.name,
                    "page_number": page["page_number"],
                    "doc_type": "reports" if "report" in pdf_path.name.lower() else "other",
                    "text_length": len(page["text"]),
                    "image_size": page["image_size"]
                }
                
                # Insert into appropriate partition
                partition_name = f"type_{metadata['doc_type']}"
                client.insert([vector], [metadata], partition_name=partition_name)
                print(f"Inserted page {page['page_number']} from {pdf_path.name}")
        
        # Perform hybrid search
        query_config = HybridQueryConfig(
            vector_field="vector",
            scalar_fields=["doc_type", "text_length"],
            metric_type="L2",
            top_k=5,
            params={"nprobe": 10}
        )
        
        # Example: Search for similar content in reports
        query_text = "Example search query about financial reports"
        query_vector = processor.generate_embedding(query_text)
        
        results = client.hybrid_search(
            [query_vector],
            query_config,
            partition_names=["type_reports"],
            doc_type="reports"
        )
        
        print("\nSearch results:")
        for i, hit in enumerate(results[0]):
            print(f"Rank {i+1}:")
            print(f"  Document: {hit['filename']}")
            print(f"  Page: {hit['page_number']}")
            print(f"  Type: {hit['doc_type']}")
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