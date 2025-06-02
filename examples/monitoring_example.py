from ai_prishtina_milvus_client import MetricsCollector, MonitoringConfig

client = ...  # Your Milvus client instance
monitoring_config = MonitoringConfig(collect_system_metrics=True)
collector = MetricsCollector(client, monitoring_config)
# system_metrics = collector.collect_system_metrics()
# print(system_metrics)
print("MetricsCollector ready for system metrics collection.") 