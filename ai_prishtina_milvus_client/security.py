"""
Security features for Milvus operations including authentication, encryption, and access control with async support.
"""

from typing import Dict, List, Optional, Union, Any, Awaitable
import os
import base64
import hashlib
import hmac
import time
import jwt
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, SecretStr
import logging
import asyncio
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import bcrypt

class SecurityConfig(BaseModel):
    """Configuration for security features."""
    secret_key: str = Field(..., description="Secret key for JWT")
    token_expiry: int = Field(3600, description="Token expiry time in seconds")
    encryption_key: str = Field(..., description="Key for data encryption")
    allowed_ips: List[str] = Field(default_factory=list, description="Allowed IP addresses")
    require_ssl: bool = Field(True, description="Whether to require SSL")

class User(BaseModel):
    """User model."""
    username: str = Field(..., description="Username")
    roles: List[str] = Field(default_factory=list, description="User roles")
    password_hash: Optional[str] = Field(None, description="Hashed password")

class SecurityManager:
    """Manager for security features."""
    
    def __init__(self, config: SecurityConfig):
        """
        Initialize security manager.
        
        Args:
            config: Security configuration
        """
        self.config = config
        self.users: Dict[str, User] = {}
        self.logger = logging.getLogger(__name__)
        
    async def create_user(self, username: str, password: str, roles: List[str]) -> None:
        """
        Create a new user asynchronously.
        
        Args:
            username: Username
            password: Password
            roles: User roles
        """
        if username in self.users:
            raise ValueError(f"User {username} already exists")
            
        password_hash = await self._hash_password(password)
        self.users[username] = User(
            username=username,
            roles=roles,
            password_hash=password_hash
        )
        
    async def authenticate(self, username: str, password: str) -> str:
        """
        Authenticate user and return JWT token asynchronously.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            JWT token
            
        Raises:
            ValueError: If authentication fails
        """
        user = self.users.get(username)
        if not user or not await self._verify_password(password, user.password_hash):
            raise ValueError("Invalid username or password")
            
        token = jwt.encode(
            {
                "username": username,
                "roles": user.roles,
                "exp": time.time() + self.config.token_expiry
            },
            self.config.secret_key,
            algorithm="HS256"
        )
        
        return token
        
    async def verify_token(self, token: str) -> User:
        """
        Verify JWT token and return user asynchronously.
        
        Args:
            token: JWT token
            
        Returns:
            User object
            
        Raises:
            ValueError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=["HS256"]
            )
            username = payload["username"]
            user = self.users.get(username)
            if not user:
                raise ValueError(f"User {username} not found")
            return user
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError:
            raise ValueError("Invalid token")
            
    async def check_permission(self, user: User, required_role: str) -> bool:
        """
        Check if user has required role asynchronously.
        
        Args:
            user: User to check
            required_role: Required role
            
        Returns:
            True if user has required role, False otherwise
        """
        return required_role in user.roles
        
    async def encrypt_data(self, data: str) -> bytes:
        """
        Encrypt data asynchronously.
        
        Args:
            data: Data to encrypt
            
        Returns:
            Encrypted data
        """
        key = self.config.encryption_key.encode()
        f = Fernet(key)
        return await asyncio.to_thread(f.encrypt, data.encode())
        
    async def decrypt_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt data asynchronously.
        
        Args:
            encrypted_data: Data to decrypt
            
        Returns:
            Decrypted data
        """
        key = self.config.encryption_key.encode()
        f = Fernet(key)
        return (await asyncio.to_thread(f.decrypt, encrypted_data)).decode()
        
    async def _hash_password(self, password: str) -> str:
        """
        Hash password asynchronously.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        return (await asyncio.to_thread(
            bcrypt.hashpw,
            password.encode(),
            bcrypt.gensalt()
        )).decode()
        
    async def _verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify password asynchronously.
        
        Args:
            password: Password to verify
            password_hash: Hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        return await asyncio.to_thread(
            bcrypt.checkpw,
            password.encode(),
            password_hash.encode()
        )
        
    async def validate_ip(self, ip: str) -> bool:
        """
        Validate IP address against allowed list asynchronously.
        
        Args:
            ip: IP address to validate
            
        Returns:
            True if IP is allowed, False otherwise
        """
        if not self.config.allowed_ips:
            return True
            
        return ip in self.config.allowed_ips 