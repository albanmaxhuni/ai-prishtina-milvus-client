from ai_prishtina_milvus_client import SecurityManager, SecurityConfig

security_config = SecurityConfig(secret_key="supersecret", token_expiry=3600)
manager = SecurityManager(security_config)
user = manager.create_user("alice", "password123", ["admin"])
token = manager.authenticate("alice", "password123")
print("JWT Token:", token) 