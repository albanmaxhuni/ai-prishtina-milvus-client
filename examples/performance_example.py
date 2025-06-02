from ai_prishtina_milvus_client import PerformanceOptimizer, CacheConfig

cache_config = CacheConfig(max_size=100, expiry_time=60)
optimizer = PerformanceOptimizer(cache_config)

@optimizer.cached
def expensive_computation(x):
    return x * x

print(expensive_computation(10)) 