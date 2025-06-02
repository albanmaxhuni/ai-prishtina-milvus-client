"""
Unit tests for security module.
"""

import pytest
from unittest.mock import Mock, patch
import time
import jwt
from ai_prishtina_milvus_client import SecurityManager, SecurityConfig, User

def test_create_user(security_config):
    """Test user creation."""
    manager = SecurityManager(security_config)
    
    # Test successful user creation
    user = manager.create_user("test_user", "password123", ["admin"])
    assert isinstance(user, User)
    assert user.username == "test_user"
    assert user.roles == ["admin"]
    assert user.is_active is True
    
    # Test duplicate username
    with pytest.raises(ValueError):
        manager.create_user("test_user", "password123", ["admin"])

def test_authenticate(security_config):
    """Test user authentication."""
    manager = SecurityManager(security_config)
    manager.create_user("test_user", "password123", ["admin"])
    
    # Test successful authentication
    token = manager.authenticate("test_user", "password123")
    assert token is not None
    decoded = jwt.decode(token, security_config.secret_key, algorithms=["HS256"])
    assert decoded["username"] == "test_user"
    assert decoded["roles"] == ["admin"]
    
    # Test invalid credentials
    with pytest.raises(ValueError):
        manager.authenticate("test_user", "wrong_password")
    
    # Test non-existent user
    with pytest.raises(ValueError):
        manager.authenticate("non_existent", "password123")

def test_verify_token(security_config):
    """Test token verification."""
    manager = SecurityManager(security_config)
    manager.create_user("test_user", "password123", ["admin"])
    token = manager.authenticate("test_user", "password123")
    
    # Test successful verification
    user = manager.verify_token(token)
    assert isinstance(user, User)
    assert user.username == "test_user"
    
    # Test expired token
    expired_token = jwt.encode(
        {
            "username": "test_user",
            "roles": ["admin"],
            "exp": time.time() - 3600
        },
        security_config.secret_key,
        algorithm="HS256"
    )
    with pytest.raises(jwt.ExpiredSignatureError):
        manager.verify_token(expired_token)
    
    # Test invalid token
    with pytest.raises(jwt.InvalidTokenError):
        manager.verify_token("invalid_token")

def test_check_permission(security_config):
    """Test permission checking."""
    manager = SecurityManager(security_config)
    manager.create_user("test_user", "password123", ["admin"])
    
    # Test successful permission check
    assert manager.check_permission("test_user", "admin") is True
    assert manager.check_permission("test_user", "user") is False
    
    # Test non-existent user
    with pytest.raises(ValueError):
        manager.check_permission("non_existent", "admin")

def test_encrypt_decrypt_data(security_config):
    """Test data encryption and decryption."""
    manager = SecurityManager(security_config)
    
    # Test successful encryption and decryption
    data = "sensitive_data"
    encrypted = manager.encrypt_data(data)
    assert encrypted != data
    decrypted = manager.decrypt_data(encrypted)
    assert decrypted == data
    
    # Test with different data types
    data = {"key": "value", "number": 123}
    encrypted = manager.encrypt_data(data)
    decrypted = manager.decrypt_data(encrypted)
    assert decrypted == data

def test_validate_ip(security_config):
    """Test IP validation."""
    manager = SecurityManager(security_config)
    
    # Test allowed IP
    security_config.allowed_ips = ["192.168.1.1"]
    assert manager.validate_ip("192.168.1.1") is True
    
    # Test disallowed IP
    assert manager.validate_ip("192.168.1.2") is False
    
    # Test with no IP restrictions
    security_config.allowed_ips = []
    assert manager.validate_ip("192.168.1.1") is True

def test_security_config():
    """Test security configuration."""
    config = SecurityConfig(
        secret_key="test_secret",
        token_expiry=3600,
        encryption_key="test_encryption_key",
        allowed_ips=["192.168.1.1"],
        require_ssl=True
    )
    
    assert config.secret_key == "test_secret"
    assert config.token_expiry == 3600
    assert config.encryption_key == "test_encryption_key"
    assert config.allowed_ips == ["192.168.1.1"]
    assert config.require_ssl is True 