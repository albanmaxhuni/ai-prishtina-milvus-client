"""
Development tools for Milvus operations including debugging, logging, and testing utilities.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable
import logging
import sys
import traceback
import time
import json
from datetime import datetime
from pathlib import Path
import inspect
import functools
import pdb
import pytest
from pydantic import BaseModel, Field
import numpy as np

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: int = Field(logging.INFO, description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_size: int = Field(10 * 1024 * 1024, description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup files")

class DebugConfig(BaseModel):
    """Configuration for debugging."""
    enabled: bool = Field(False, description="Whether debugging is enabled")
    break_on_error: bool = Field(False, description="Whether to break on error")
    log_level: int = Field(logging.DEBUG, description="Debug log level")
    trace_calls: bool = Field(False, description="Whether to trace function calls")
    level: int = Field(logging.DEBUG, description="Debug log level (alias)")

class TestConfig(BaseModel):
    """Configuration for testing."""
    test_dir: str = Field("tests", description="Test directory")
    file_pattern: str = Field("test_*.py", description="Test file pattern")
    collect_coverage: bool = Field(True, description="Whether to collect coverage")
    parallel_execution: bool = Field(False, description="Whether to run tests in parallel")
    level: int = Field(logging.INFO, description="Logging level for tests")

class DevTools:
    """Development tools for Milvus operations."""
    
    def __init__(
        self,
        logging_config: Optional[LoggingConfig] = None,
        debug_config: Optional[DebugConfig] = None,
        test_config: Optional[TestConfig] = None
    ):
        """
        Initialize development tools.
        
        Args:
            logging_config: Optional logging configuration
            debug_config: Optional debug configuration
            test_config: Optional test configuration
        """
        self.logging_config = logging_config or LoggingConfig()
        self.debug_config = debug_config or DebugConfig()
        self.test_config = test_config or TestConfig()
        
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(self.logging_config.level)
        
        # Create formatter
        formatter = logging.Formatter(getattr(self.logging_config, 'format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Create file handler if configured
        if self.logging_config.file_path:
            from logging.handlers import RotatingFileHandler
            
            file_handler = RotatingFileHandler(
                self.logging_config.file_path,
                maxBytes=self.logging_config.max_size,
                backupCount=self.logging_config.backup_count
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
    def debug(self, func: Callable) -> Callable:
        """
        Decorator to add debugging capabilities.
        
        Args:
            func: Function to debug
            
        Returns:
            Wrapped function with debugging
        """
        if not self.debug_config.enabled:
            return func
            
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log function call
            if self.debug_config.trace_calls:
                self.logger.debug(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                
            try:
                # Call function
                result = func(*args, **kwargs)
                
                # Log result
                if self.debug_config.trace_calls:
                    self.logger.debug(
                        f"{func.__name__} returned: {result}"
                    )
                    
                return result
                
            except Exception as e:
                # Log error
                self.logger.error(
                    f"Error in {func.__name__}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                
                # Break on error if configured
                if self.debug_config.break_on_error:
                    pdb.set_trace()
                    
                raise
                
        return wrapper
        
    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start timer
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Call function
            result = func(*args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            metrics = {
                "function": func.__name__,
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log metrics
            self.logger.debug(f"Profile metrics: {json.dumps(metrics, indent=2)}")
            
            return result
            
        return wrapper
        
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage.
        
        Returns:
            Memory usage in bytes
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
        
    def run_tests(self) -> Tuple[int, List[str]]:
        """
        Run tests.
        
        Returns:
            Tuple of (exit code, test output)
        """
        import subprocess
        
        # Build pytest command
        cmd = ["pytest", self.test_config.test_dir]
        
        if self.test_config.file_pattern:
            cmd.extend(["-k", self.test_config.file_pattern])
            
        if self.test_config.collect_coverage:
            cmd.extend(["--cov=ai_prishtina_milvus_client"])
            
        if self.test_config.parallel_execution:
            cmd.extend(["-n", "auto"])
            
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            return result.returncode, result.stdout.splitlines()
            
        except Exception as e:
            self.logger.error(f"Failed to run tests: {str(e)}")
            raise
            
    def generate_test_data(
        self,
        num_vectors: int,
        vector_dim: int,
        metadata_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate test data.
        
        Args:
            num_vectors: Number of vectors to generate
            vector_dim: Vector dimension
            metadata_fields: Optional list of metadata field names
            
        Returns:
            List of test records
        """
        data = []
        
        for i in range(num_vectors):
            record = {
                "id": i,
                "vector": np.random.rand(vector_dim).tolist()
            }
            
            if metadata_fields:
                record["metadata"] = {
                    field: f"value_{i}_{j}"
                    for j, field in enumerate(metadata_fields)
                }
                
            data.append(record)
            
        return data
        
    def validate_test_results(
        self,
        expected: List[Dict[str, Any]],
        actual: List[Dict[str, Any]],
        tolerance: float = 1e-6
    ) -> Tuple[bool, List[str]]:
        """
        Validate test results.
        
        Args:
            expected: Expected results
            actual: Actual results
            tolerance: Tolerance for floating point comparison
            
        Returns:
            Tuple of (is_valid, error messages)
        """
        errors = []
        
        # Check length
        if len(expected) != len(actual):
            errors.append(
                f"Length mismatch: expected {len(expected)}, got {len(actual)}"
            )
            return False, errors
            
        # Check each record
        for i, (exp, act) in enumerate(zip(expected, actual)):
            # Check ID
            if exp["id"] != act["id"]:
                errors.append(
                    f"Record {i}: ID mismatch: expected {exp['id']}, got {act['id']}"
                )
                
            # Check vector
            if "vector" in exp:
                exp_vec = np.array(exp["vector"])
                act_vec = np.array(act["vector"])
                
                if not np.allclose(exp_vec, act_vec, rtol=tolerance):
                    errors.append(
                        f"Record {i}: Vector mismatch: expected {exp_vec}, got {act_vec}"
                    )
                    
            # Check metadata
            if "metadata" in exp:
                exp_meta = exp["metadata"]
                act_meta = act.get("metadata", {})
                
                if exp_meta != act_meta:
                    errors.append(
                        f"Record {i}: Metadata mismatch: expected {exp_meta}, got {act_meta}"
                    )
                    
            # Check numeric fields with tolerance
            for key in exp:
                if key not in ["id", "vector", "metadata"]:
                    if isinstance(exp[key], (int, float)):
                        if not np.isclose(exp[key], act[key], rtol=tolerance):
                            errors.append(
                                f"Record {i}: {key} mismatch: expected {exp[key]}, got {act[key]}"
                            )
                    elif exp[key] != act[key]:
                        errors.append(
                            f"Record {i}: {key} mismatch: expected {exp[key]}, got {act[key]}"
                        )
                    
        return len(errors) == 0, errors
        
    def create_test_collection(
        self,
        client: Any,
        collection_name: str,
        vector_dim: int,
        num_vectors: int,
        num_metadata: int = 0
    ) -> None:
        """
        Create test collection with data.
        
        Args:
            client: Milvus client
            collection_name: Collection name
            vector_dim: Vector dimension
            num_vectors: Number of vectors
            num_metadata: Number of metadata fields
        """
        # Generate test data
        data = self.generate_test_data(
            num_vectors,
            vector_dim,
            num_metadata
        )
        
        # Create collection
        fields = [
            {"name": "id", "dtype": "INT64", "is_primary": True},
            {"name": "vector", "dtype": "FLOAT_VECTOR", "dim": vector_dim}
        ]
        
        if num_metadata > 0:
            fields.append(
                {"name": "metadata", "dtype": "JSON"}
            )
            
        client.create_collection(
            collection_name=collection_name,
            fields=fields
        )
        
        # Insert data
        client.insert(
            collection_name=collection_name,
            data=data
        )
        
    def cleanup_test_collection(
        self,
        client: Any,
        collection_name: str
    ) -> None:
        """
        Clean up test collection.
        
        Args:
            client: Milvus client
            collection_name: Collection name
        """
        try:
            client.drop_collection(collection_name)
        except Exception as e:
            self.logger.warning(f"Failed to drop collection {collection_name}: {str(e)}")
            
    def get_function_info(self, func: Callable) -> Dict[str, Any]:
        """
        Get function information.
        
        Args:
            func: Function to inspect
            
        Returns:
            Function information
        """
        info = {
            "name": func.__name__,
            "module": func.__module__,
            "docstring": func.__doc__,
            "signature": str(inspect.signature(func)),
            "source": inspect.getsource(func),
            "is_async": inspect.iscoroutinefunction(func),
            "is_generator": inspect.isgeneratorfunction(func)
        }
        
        return info 