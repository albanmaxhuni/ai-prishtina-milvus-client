"""
Error recovery and retry mechanisms for Milvus operations.
"""

from typing import TypeVar, Callable, Optional, List, Dict, Any, Union
import time
import random
from functools import wraps
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

T = TypeVar('T')

class RetryConfig(BaseModel):
    """Configuration for retry mechanism."""
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    exponential_backoff: bool = Field(True, description="Whether to use exponential backoff")
    max_delay: float = Field(10.0, description="Maximum delay between retries in seconds")
    retry_on_exceptions: List[type] = Field(
        default_factory=lambda: [Exception],
        description="List of exceptions to retry on"
    )

class BackupConfig(BaseModel):
    """Configuration for backup operations."""
    backup_dir: str = Field(..., description="Directory to store backups")
    max_backups: int = Field(5, description="Maximum number of backups to keep")
    backup_interval: float = Field(3600.0, description="Backup interval in seconds")
    compression: bool = Field(True, description="Whether to compress backups")

class ErrorRecovery:
    """Utility class for error recovery and retry mechanisms."""
    
    def __init__(
        self,
        client,
        retry_config: Optional[RetryConfig] = None,
        backup_config: Optional[BackupConfig] = None
    ):
        """
        Initialize error recovery.
        
        Args:
            client: Milvus client
            retry_config: Optional retry configuration
            backup_config: Optional backup configuration
        """
        self.client = client
        self.retry_config = retry_config or RetryConfig()
        self.backup_config = backup_config
        self.logger = logging.getLogger(__name__)
        
    def retry_operation(self, func: Callable, *args, **kwargs):
        """
        Retry an operation with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except tuple(self.retry_config.retry_on_exceptions) as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.retry_delay
                    if self.retry_config.exponential_backoff:
                        delay = min(delay * (2 ** attempt), self.retry_config.max_delay)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.retry_config.max_retries} attempts failed. Last error: {str(e)}")
                    raise last_exception
        raise last_exception
        
    def create_backup(self, collection_name: str) -> str:
        """
        Create a backup of a collection.
        
        Args:
            collection_name: Name of collection to backup
            
        Returns:
            Path to backup file
            
        Raises:
            ValueError: If backup configuration is not provided
        """
        if not self.backup_config:
            raise ValueError("Backup configuration is required")
            
        import os
        import json
        import shutil
        from datetime import datetime
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_config.backup_dir, exist_ok=True)
        
        # Generate backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{collection_name}_{timestamp}"
        backup_path = os.path.join(self.backup_config.backup_dir, backup_name)
        
        try:
            # Create backup
            if self.backup_config.compression:
                backup_path += ".tar.gz"
                with tarfile.open(backup_path, "w:gz") as tar:
                    tar.add(collection_name)
            else:
                shutil.copytree(collection_name, backup_path)
                
            # Clean up old backups
            self._cleanup_old_backups(collection_name)
            
            return backup_path
            
        except Exception as e:
            self.logger.error(f"Failed to create backup: {str(e)}")
            raise
            
    def restore_backup(self, backup_path: str, collection_name: str) -> None:
        """
        Restore a collection from backup.
        
        Args:
            backup_path: Path to backup file
            collection_name: Name of collection to restore
            
        Raises:
            ValueError: If backup configuration is not provided
        """
        if not self.backup_config:
            raise ValueError("Backup configuration is required")
            
        import os
        import shutil
        import tarfile
        
        try:
            # Remove existing collection if it exists
            if os.path.exists(collection_name):
                shutil.rmtree(collection_name)
                
            # Restore from backup
            if backup_path.endswith(".tar.gz"):
                with tarfile.open(backup_path, "r:gz") as tar:
                    tar.extractall()
            else:
                shutil.copytree(backup_path, collection_name)
                
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {str(e)}")
            raise
            
    def _cleanup_old_backups(self, collection_name: str) -> None:
        """
        Clean up old backups.
        
        Args:
            collection_name: Name of collection
        """
        import os
        import glob
        
        # Get list of backups for collection
        pattern = os.path.join(self.backup_config.backup_dir, f"{collection_name}_*")
        backups = sorted(glob.glob(pattern))
        
        # Remove old backups
        while len(backups) > self.backup_config.max_backups:
            os.remove(backups.pop(0))
            
    def auto_backup(self, collection_name: str) -> None:
        """
        Set up automatic backup for a collection.
        
        Args:
            collection_name: Name of collection to backup
        """
        if not self.backup_config:
            raise ValueError("Backup configuration is required")
            
        import threading
        
        def backup_loop():
            while True:
                try:
                    self.create_backup(collection_name)
                except Exception as e:
                    self.logger.error(f"Automatic backup failed: {str(e)}")
                time.sleep(self.backup_config.backup_interval)
                
        # Start backup thread
        thread = threading.Thread(target=backup_loop, daemon=True)
        thread.start()
        
    def reconnect(self, connect_func: Callable[[], None]) -> None:
        """
        Attempt to reconnect with exponential backoff.
        
        Args:
            connect_func: Function to call for reconnection
        """
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                connect_func()
                return
            except Exception as e:
                if attempt < self.retry_config.max_retries:
                    delay = self.retry_config.retry_delay
                    if self.retry_config.exponential_backoff:
                        delay = min(delay * (2 ** attempt), self.retry_config.max_delay)
                    self.logger.warning(
                        f"Reconnection attempt {attempt + 1} failed: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.retry_config.max_retries} reconnection attempts failed. "
                        f"Last error: {str(e)}"
                    )
                    raise 

    def backup_collection(self, collection_name: str) -> str:
        """
        Create a backup of a collection.
        
        Args:
            collection_name: Name of collection to backup
            
        Returns:
            Path to backup file
            
        Raises:
            ValueError: If backup configuration is not provided
        """
        if not self.backup_config:
            raise ValueError("Backup configuration is required")
            
        return self.create_backup(collection_name)
        
    def restore_collection(self, collection_name: str, backup_path: str) -> None:
        """
        Restore a collection from backup.
        
        Args:
            collection_name: Name of collection to restore
            backup_path: Path to backup file
            
        Raises:
            ValueError: If backup configuration is not provided
        """
        if not self.backup_config:
            raise ValueError("Backup configuration is required")
            
        return self.restore_backup(backup_path, collection_name)
        
    def handle_connection_error(self) -> bool:
        """
        Handle connection error by attempting to reconnect.
        
        Returns:
            True if reconnection successful, False otherwise
        """
        try:
            self.client.connect()
            return True
        except Exception as e:
            self.logger.error(f"Failed to reconnect: {e}")
            return False 