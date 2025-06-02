from ai_prishtina_milvus_client import AdvancedSearch, SearchConfig

client = ...  # Your Milvus client instance
search_config = SearchConfig(metric_type="L2", top_k=5)
search = AdvancedSearch(client, search_config)
vectors = [[0.1] * 128]
# results = search.hybrid_search(vectors=vectors, text_query="example")
# print(results)
print("AdvancedSearch ready for hybrid search.") 