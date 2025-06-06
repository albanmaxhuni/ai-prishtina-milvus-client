"""
Error recovery and retry mechanisms with async support.
"""

import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union, Awaitable
from datetime import datetime, timedelta
import traceback

from pydantic import BaseModel, Field

from .exceptions import ErrorRecoveryError
from .client import AsyncMilvusClient
from .config import MilvusConfig


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = Field(3, description="Maximum number of retry attempts")
    initial_delay: float = Field(1.0, description="Initial delay between retries in seconds")
    max_delay: float = Field(30.0, description="Maximum delay between retries in seconds")
    backoff_factor: float = Field(2.0, description="Exponential backoff factor")
    retry_on_exceptions: List[Type[Exception]] = Field(
        default=[Exception],
        description="List of exceptions to retry on"
    )


class ErrorRecovery:
    """Error recovery and retry mechanisms for async operations."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        retry_config: Optional[RetryConfig] = None,
        client: Optional[AsyncMilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.retry_config = retry_config or RetryConfig()
        self.client = client or AsyncMilvusClient(milvus_config)
        self.logger = logging.getLogger(__name__)
        
    async def retry_operation(
        self,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Retry an async operation with exponential backoff.
        
        Args:
            operation: Async operation to retry
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            ErrorRecoveryError: If all retry attempts fail
        """
        last_exception = None
        delay = self.retry_config.initial_delay
        
        for attempt in range(self.retry_config.max_attempts):
            try:
                return await operation(*args, **kwargs)
            except tuple(self.retry_config.retry_on_exceptions) as e:
                last_exception = e
                if attempt < self.retry_config.max_attempts - 1:
                    self.logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.retry_config.max_attempts}): {str(e)}"
                    )
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * self.retry_config.backoff_factor,
                        self.retry_config.max_delay
                    )
                else:
                    self.logger.error(
                        f"Operation failed after {self.retry_config.max_attempts} attempts: {str(e)}"
                    )
                    
        raise ErrorRecoveryError(
            f"Operation failed after {self.retry_config.max_attempts} attempts: {str(last_exception)}"
        )
        
    async def recover_connection(self) -> bool:
        """
        Attempt to recover the Milvus connection.
        
        Returns:
            bool: True if recovery was successful, False otherwise
        """
        try:
            await self.client.connect()
            return True
        except Exception as e:
            self.logger.error(f"Failed to recover connection: {str(e)}")
            return False
            
    async def handle_operation_error(
        self,
        operation: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Handle operation errors with recovery mechanisms.
        
        Args:
            operation: Async operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            ErrorRecoveryError: If recovery attempts fail
        """
        try:
            return await self.retry_operation(operation, *args, **kwargs)
        except Exception as e:
            self.logger.error(f"Operation failed: {str(e)}")
            
            # Attempt connection recovery
            if await self.recover_connection():
                try:
                    return await self.retry_operation(operation, *args, **kwargs)
                except Exception as retry_e:
                    raise ErrorRecoveryError(
                        f"Operation failed after recovery: {str(retry_e)}"
                    )
            else:
                raise ErrorRecoveryError(
                    f"Operation failed and recovery unsuccessful: {str(e)}"
                )
                
    async def cleanup_resources(self) -> None:
        """Cleanup resources and connections."""
        try:
            await self.client.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    async def __aenter__(self):
        """Context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self.cleanup_resources() 