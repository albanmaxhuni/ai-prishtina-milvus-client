from ai_prishtina_milvus_client import DevTools, LoggingConfig

logging_config = LoggingConfig(level="INFO", file_path="dev.log")
tools = DevTools(logging_config)

@tools.debug
def add(a, b):
    return a + b

print(add(2, 3)) 