"""
Data management utilities for Milvus with async support.
"""

import json
import os
from typing import Any, Dict, List, Optional, Union, Awaitable
from datetime import datetime
import asyncio
import aiofiles
from pydantic import BaseModel, Field

from .client import AsyncMilvusClient
from .config import MilvusConfig
from .exceptions import DataManagementError


class DataManagementConfig(BaseModel):
    """Configuration for data management."""
    backup_dir: str = Field(..., description="Directory for backups")
    export_dir: str = Field(..., description="Directory for exports")
    import_dir: str = Field(..., description="Directory for imports")
    max_backups: int = Field(5, description="Maximum number of backups to keep")
    compression: bool = Field(True, description="Enable compression for backups")
    validate_on_import: bool = Field(True, description="Validate data on import")
    batch_size: int = Field(1000, description="Batch size for operations")


class DataManager:
    """Data management utilities for Milvus."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        data_config: DataManagementConfig,
        client: Optional[AsyncMilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.data_config = data_config
        self.client = client or AsyncMilvusClient(milvus_config)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        for directory in [
            self.data_config.backup_dir,
            self.data_config.export_dir,
            self.data_config.import_dir
        ]:
            os.makedirs(directory, exist_ok=True)
            
    async def create_backup(self, collection_name: str) -> str:
        """Create a backup of collection data asynchronously."""
        try:
            # Get collection data
            collection = await self.client.get_collection(collection_name)
            schema = await collection.schema
            data = await collection.query(expr="id >= 0")
            
            # Create backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(
                self.data_config.backup_dir,
                f"{collection_name}_{timestamp}.json"
            )
            
            # Save backup
            backup_data = {
                "schema": schema.to_dict(),
                "data": data
            }
            
            async with aiofiles.open(backup_file, "w") as f:
                await f.write(json.dumps(backup_data, indent=2))
                
            # Cleanup old backups
            await self._cleanup_old_backups(collection_name)
            
            return backup_file
            
        except Exception as e:
            raise DataManagementError(f"Failed to create backup: {str(e)}")
            
    async def restore_backup(self, backup_file: str) -> None:
        """Restore collection from backup asynchronously."""
        try:
            # Read backup file
            async with aiofiles.open(backup_file, "r") as f:
                backup_data = json.loads(await f.read())
                
            # Create collection if not exists
            collection_name = os.path.basename(backup_file).split("_")[0]
            if not await self.client.has_collection(collection_name):
                await self.client.create_collection(
                    collection_name,
                    backup_data["schema"]
                )
                
            # Insert data
            collection = await self.client.get_collection(collection_name)
            await collection.insert(backup_data["data"])
            
        except Exception as e:
            raise DataManagementError(f"Failed to restore backup: {str(e)}")
            
    async def export_data(
        self,
        collection_name: str,
        output_format: str = "json"
    ) -> str:
        """Export collection data asynchronously."""
        try:
            # Get collection data
            collection = await self.client.get_collection(collection_name)
            data = await collection.query(expr="id >= 0")
            
            # Create export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = os.path.join(
                self.data_config.export_dir,
                f"{collection_name}_{timestamp}.{output_format}"
            )
            
            # Export data
            if output_format == "json":
                async with aiofiles.open(export_file, "w") as f:
                    await f.write(json.dumps(data, indent=2))
            else:
                raise DataManagementError(f"Unsupported export format: {output_format}")
                
            return export_file
            
        except Exception as e:
            raise DataManagementError(f"Failed to export data: {str(e)}")
            
    async def import_data(
        self,
        import_file: str,
        collection_name: Optional[str] = None
    ) -> None:
        """Import data into collection asynchronously."""
        try:
            # Read import file
            async with aiofiles.open(import_file, "r") as f:
                data = json.loads(await f.read())
                
            # Determine collection name
            if collection_name is None:
                collection_name = os.path.basename(import_file).split("_")[0]
                
            # Validate data if enabled
            if self.data_config.validate_on_import:
                await self._validate_import_data(data)
                
            # Insert data in batches
            collection = await self.client.get_collection(collection_name)
            for i in range(0, len(data), self.data_config.batch_size):
                batch = data[i:i + self.data_config.batch_size]
                await collection.insert(batch)
                
        except Exception as e:
            raise DataManagementError(f"Failed to import data: {str(e)}")
            
    async def _validate_import_data(self, data: List[Dict[str, Any]]) -> None:
        """Validate imported data asynchronously."""
        if not isinstance(data, list):
            raise DataManagementError("Import data must be a list")
            
        if not data:
            raise DataManagementError("Import data is empty")
            
        # Validate each record
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                raise DataManagementError(f"Record {i} must be a dictionary")
                
            # Add more validation as needed
            
    async def _cleanup_old_backups(self, collection_name: str) -> None:
        """Cleanup old backups asynchronously."""
        try:
            # Get backup files
            backup_files = [
                f for f in os.listdir(self.data_config.backup_dir)
                if f.startswith(f"{collection_name}_")
            ]
            
            # Sort by timestamp
            backup_files.sort(reverse=True)
            
            # Remove old backups
            for old_file in backup_files[self.data_config.max_backups:]:
                os.remove(os.path.join(self.data_config.backup_dir, old_file))
                
        except Exception as e:
            raise DataManagementError(f"Failed to cleanup old backups: {str(e)}")
            
    async def get_backup_info(self, collection_name: str) -> List[Dict[str, Any]]:
        """Get information about backups asynchronously."""
        try:
            backup_files = [
                f for f in os.listdir(self.data_config.backup_dir)
                if f.startswith(f"{collection_name}_")
            ]
            
            backup_info = []
            for backup_file in backup_files:
                file_path = os.path.join(self.data_config.backup_dir, backup_file)
                stat = os.stat(file_path)
                backup_info.append({
                    "filename": backup_file,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat()
                })
                
            return sorted(backup_info, key=lambda x: x["created"], reverse=True)
            
        except Exception as e:
            raise DataManagementError(f"Failed to get backup info: {str(e)}")
            
    async def cleanup_export_dir(self) -> None:
        """Cleanup export directory asynchronously."""
        try:
            for file in os.listdir(self.data_config.export_dir):
                os.remove(os.path.join(self.data_config.export_dir, file))
        except Exception as e:
            raise DataManagementError(f"Failed to cleanup export directory: {str(e)}")
            
    async def cleanup_import_dir(self) -> None:
        """Cleanup import directory asynchronously."""
        try:
            for file in os.listdir(self.data_config.import_dir):
                os.remove(os.path.join(self.data_config.import_dir, file))
        except Exception as e:
            raise DataManagementError(f"Failed to cleanup import directory: {str(e)}") 