from ai_prishtina_milvus_client import ErrorRecovery, RetryConfig

client = ...  # Your Milvus client instance
retry_config = RetryConfig(max_retries=3, initial_delay=1.0, max_delay=10.0)
recovery = ErrorRecovery(client, retry_config)

@recovery.with_retry
def unreliable_operation():
    # Some operation that may fail
    pass

# unreliable_operation()
print("ErrorRecovery ready for retry operations.") 