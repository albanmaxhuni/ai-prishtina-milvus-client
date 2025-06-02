from ai_prishtina_milvus_client import BatchProcessor, BatchConfig

# Example: Batch insert vectors
client = ...  # Your Milvus client instance
batch_config = BatchConfig(batch_size=100, max_workers=4, show_progress=True)
processor = BatchProcessor(client, batch_config)
vectors = [[0.1] * 128 for _ in range(1000)]
metadata = [{"id": i} for i in range(1000)]
# metrics = processor.batch_insert(vectors, metadata)
# print(metrics)
print("BatchProcessor ready for batch_insert.") 