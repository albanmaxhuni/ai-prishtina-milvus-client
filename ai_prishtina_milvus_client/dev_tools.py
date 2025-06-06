"""
Development tools for Milvus operations including debugging, logging, and testing utilities with async support.
"""

from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Awaitable
import logging
import sys
import traceback
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
import inspect
import functools
import pdb
import pytest
import asyncio
from pydantic import BaseModel, Field
import numpy as np

from .exceptions import DevToolsError
from .client import AsyncMilvusClient
from .config import MilvusConfig

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: int = Field(logging.INFO, description="Logging level")
    format: str = Field("%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    file_path: Optional[str] = Field(None, description="Log file path")
    max_size: int = Field(10 * 1024 * 1024, description="Maximum log file size")
    backup_count: int = Field(5, description="Number of backup files")

class DebugConfig(BaseModel):
    """Configuration for debugging."""
    enabled: bool = Field(True, description="Whether debugging is enabled")
    break_on_error: bool = Field(False, description="Whether to break on error")
    log_level: str = Field("DEBUG", description="Logging level")
    trace_calls: bool = Field(True, description="Whether to trace function calls")
    trace_operations: bool = Field(True, description="Whether to trace operations")
    profile_operations: bool = Field(False, description="Whether to profile operations")
    max_trace_size: int = Field(1000, description="Maximum number of traced operations")
    retention_period: float = Field(3600.0, description="Trace retention period in seconds")

class TestConfig(BaseModel):
    """Configuration for testing."""
    test_dir: str = Field("tests", description="Test directory")
    file_pattern: str = Field("test_*.py", description="Test file pattern")
    collect_coverage: bool = Field(True, description="Whether to collect coverage")
    parallel_execution: bool = Field(False, description="Whether to run tests in parallel")
    level: int = Field(logging.INFO, description="Logging level for tests")

class ProfilerConfig(BaseModel):
    """Configuration for profiling."""
    enabled: bool = Field(False, description="Whether profiling is enabled")
    sample_rate: float = Field(0.1, description="Sampling rate for profiling")
    max_samples: int = Field(1000, description="Maximum number of samples")
    metrics: List[str] = Field(
        default=["time", "memory", "cpu"],
        description="Metrics to collect"
    )

class DevTools:
    """Development tools for Milvus operations."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        logging_config: Optional[LoggingConfig] = None,
        debug_config: Optional[DebugConfig] = None,
        test_config: Optional[TestConfig] = None,
        profiler_config: Optional[ProfilerConfig] = None,
        client: Optional[AsyncMilvusClient] = None
    ):
        """
        Initialize development tools.
        
        Args:
            milvus_config: Milvus configuration
            logging_config: Optional logging configuration
            debug_config: Optional debug configuration
            test_config: Optional test configuration
            profiler_config: Optional profiling configuration
            client: Optional Milvus client
        """
        self.milvus_config = milvus_config
        self.logging_config = logging_config or LoggingConfig()
        self.debug_config = debug_config or DebugConfig()
        self.test_config = test_config or TestConfig()
        self.profiler_config = profiler_config or ProfilerConfig()
        self.client = client or AsyncMilvusClient(milvus_config)
        
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        self._traces: List[Dict[str, Any]] = []
        self._profiles: Dict[str, List[Dict[str, Any]]] = {}
        self._is_running = False
        
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
        async def async_wrapper(*args, **kwargs):
            # Log function call
            if self.debug_config.trace_calls:
                self.logger.debug(
                    f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}"
                )
                
            try:
                # Call function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(func, *args, **kwargs)
                
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
                
        return async_wrapper
        
    def profile(self, func: Callable) -> Callable:
        """
        Decorator to profile function execution.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapped function with profiling
        """
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Start timer
            start_time = time.time()
            start_memory = await asyncio.to_thread(self._get_memory_usage)
            
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.to_thread(func, *args, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = await asyncio.to_thread(self._get_memory_usage)
            
            metrics = {
                "function": func.__name__,
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log metrics
            self.logger.debug(f"Profile metrics: {json.dumps(metrics, indent=2)}")
            
            return result
            
        return async_wrapper
        
    def _get_memory_usage(self) -> float:
        """
        Get current memory usage.
        
        Returns:
            Memory usage in bytes
        """
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
        
    async def run_tests(self) -> Tuple[int, List[str]]:
        """
        Run tests asynchronously.
        
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
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return process.returncode, stdout.decode().splitlines()
            
        except Exception as e:
            self.logger.error(f"Failed to run tests: {str(e)}")
            raise
            
    async def generate_test_data(
        self,
        num_vectors: int,
        vector_dim: int,
        metadata_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate test data asynchronously.
        
        Args:
            num_vectors: Number of vectors to generate
            vector_dim: Vector dimension
            metadata_fields: Optional list of metadata field names
            
        Returns:
            List of test records
        """
        async def generate_record(i: int) -> Dict[str, Any]:
            record = {
                "id": i,
                "vector": (await asyncio.to_thread(np.random.rand, vector_dim)).tolist()
            }
            
            if metadata_fields:
                record["metadata"] = {
                    field: f"value_{i}_{j}"
                    for j, field in enumerate(metadata_fields)
                }
                
            return record
            
        # Generate records concurrently
        tasks = [generate_record(i) for i in range(num_vectors)]
        return await asyncio.gather(*tasks)
        
    async def validate_test_results(
        self,
        expected: List[Dict[str, Any]],
        actual: List[Dict[str, Any]],
        tolerance: float = 1e-6
    ) -> Tuple[bool, List[str]]:
        """
        Validate test results asynchronously.
        
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
        async def validate_record(i: int, exp: Dict[str, Any], act: Dict[str, Any]) -> List[str]:
            record_errors = []
            
            # Check ID
            if exp["id"] != act["id"]:
                record_errors.append(
                    f"Record {i}: ID mismatch: expected {exp['id']}, got {act['id']}"
                )
                
            # Check vector
            if "vector" in exp:
                exp_vec = np.array(exp["vector"])
                act_vec = np.array(act["vector"])
                
                if not await asyncio.to_thread(np.allclose, exp_vec, act_vec, rtol=tolerance):
                    record_errors.append(
                        f"Record {i}: Vector mismatch: expected {exp_vec}, got {act_vec}"
                    )
                    
            # Check metadata
            if "metadata" in exp:
                exp_meta = exp["metadata"]
                act_meta = act.get("metadata", {})
                
                if exp_meta != act_meta:
                    record_errors.append(
                        f"Record {i}: Metadata mismatch: expected {exp_meta}, got {act_meta}"
                    )
                    
            # Check numeric fields with tolerance
            for key in exp:
                if key not in ["id", "vector", "metadata"]:
                    if isinstance(exp[key], (int, float)):
                        if not await asyncio.to_thread(np.isclose, exp[key], act[key], rtol=tolerance):
                            record_errors.append(
                                f"Record {i}: {key} mismatch: expected {exp[key]}, got {act[key]}"
                            )
                    elif exp[key] != act[key]:
                        record_errors.append(
                            f"Record {i}: {key} mismatch: expected {exp[key]}, got {act[key]}"
                        )
                        
            return record_errors
            
        # Validate records concurrently
        tasks = [validate_record(i, exp, act) for i, (exp, act) in enumerate(zip(expected, actual))]
        record_errors = await asyncio.gather(*tasks)
        
        # Flatten errors
        for errors_list in record_errors:
            errors.extend(errors_list)
                    
        return len(errors) == 0, errors
        
    async def create_test_collection(
        self,
        client: Any,
        collection_name: str,
        vector_dim: int,
        num_vectors: int,
        num_metadata: int = 0
    ) -> None:
        """
        Create test collection with data asynchronously.
        
        Args:
            client: Milvus client
            collection_name: Collection name
            vector_dim: Vector dimension
            num_vectors: Number of vectors
            num_metadata: Number of metadata fields
        """
        # Generate test data
        data = await self.generate_test_data(
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
            
        await client.create_collection(
            collection_name=collection_name,
            fields=fields
        )
        
        # Insert data
        await client.insert(
            collection_name=collection_name,
            data=data
        )
        
    async def cleanup_test_collection(
        self,
        client: Any,
        collection_name: str
    ) -> None:
        """
        Clean up test collection asynchronously.
        
        Args:
            client: Milvus client
            collection_name: Collection name
        """
        try:
            await client.drop_collection(collection_name)
        except Exception as e:
            self.logger.warning(f"Failed to drop collection {collection_name}: {str(e)}")
            
    async def get_function_info(self, func: Callable) -> Dict[str, Any]:
        """
        Get function information asynchronously.
        
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
            "source": await asyncio.to_thread(inspect.getsource, func),
            "is_async": inspect.iscoroutinefunction(func),
            "is_generator": inspect.isgeneratorfunction(func)
        }
        
        return info 

    async def start(self) -> None:
        """Start development tools."""
        try:
            if self._is_running:
                return
                
            self._is_running = True
            
            # Configure logging
            logging.basicConfig(level=getattr(logging, self.debug_config.log_level))
            
            # Start profiling if enabled
            if self.profiler_config.enabled:
                await self._start_profiling()
                
        except Exception as e:
            raise DevToolsError(f"Failed to start development tools: {str(e)}")
            
    async def stop(self) -> None:
        """Stop development tools."""
        try:
            if not self._is_running:
                return
                
            self._is_running = False
            
            # Stop profiling if enabled
            if self.profiler_config.enabled:
                await self._stop_profiling()
                
            # Cleanup resources
            await self.cleanup()
            
        except Exception as e:
            raise DevToolsError(f"Failed to stop development tools: {str(e)}")
            
    async def trace_operation(
        self,
        operation: str,
        args: Optional[Dict[str, Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        result: Optional[Any] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        Trace an operation.
        
        Args:
            operation: Operation name
            args: Positional arguments
            kwargs: Keyword arguments
            result: Operation result
            error: Operation error if any
            
        Raises:
            DevToolsError: If tracing fails
        """
        try:
            if not self.debug_config.trace_operations:
                return
                
            trace = {
                "timestamp": datetime.now(),
                "operation": operation,
                "args": args or {},
                "kwargs": kwargs or {},
                "result": result,
                "error": str(error) if error else None,
                "stack_trace": traceback.format_exc() if error else None
            }
            
            self._traces.append(trace)
            
            # Trim traces if needed
            if len(self._traces) > self.debug_config.max_trace_size:
                self._traces = self._traces[-self.debug_config.max_trace_size:]
                
        except Exception as e:
            raise DevToolsError(f"Failed to trace operation: {str(e)}")
            
    async def get_traces(
        self,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get operation traces.
        
        Args:
            operation: Optional operation name to filter by
            start_time: Optional start time for traces
            end_time: Optional end time for traces
            
        Returns:
            List of trace data
            
        Raises:
            DevToolsError: If trace retrieval fails
        """
        try:
            traces = self._traces
            
            # Filter by operation
            if operation:
                traces = [t for t in traces if t["operation"] == operation]
                
            # Filter by time range
            if start_time or end_time:
                traces = [
                    t for t in traces
                    if (not start_time or t["timestamp"] >= start_time) and
                       (not end_time or t["timestamp"] <= end_time)
                ]
                
            return traces
            
        except Exception as e:
            raise DevToolsError(f"Failed to get traces: {str(e)}")
            
    async def _start_profiling(self) -> None:
        """Start profiling."""
        try:
            # Initialize profiling data
            for metric in self.profiler_config.metrics:
                self._profiles[metric] = []
                
        except Exception as e:
            raise DevToolsError(f"Failed to start profiling: {str(e)}")
            
    async def _stop_profiling(self) -> None:
        """Stop profiling."""
        try:
            # Cleanup profiling data
            self._profiles.clear()
            
        except Exception as e:
            raise DevToolsError(f"Failed to stop profiling: {str(e)}")
            
    async def profile_operation(
        self,
        operation: str,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        """
        Profile an operation.
        
        Args:
            operation: Operation name
            func: Async function to profile
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            DevToolsError: If profiling fails
        """
        try:
            if not self.profiler_config.enabled:
                return await func(*args, **kwargs)
                
            # Sample operation
            if asyncio.get_event_loop().time() % (1 / self.profiler_config.sample_rate) > 1:
                return await func(*args, **kwargs)
                
            # Collect metrics
            start_time = datetime.now()
            start_memory = await self._get_memory_usage()
            start_cpu = await self._get_cpu_usage()
            
            try:
                result = await func(*args, **kwargs)
                
                # Record metrics
                end_time = datetime.now()
                end_memory = await self._get_memory_usage()
                end_cpu = await self._get_cpu_usage()
                
                profile = {
                    "timestamp": end_time,
                    "operation": operation,
                    "duration": (end_time - start_time).total_seconds(),
                    "memory_delta": end_memory - start_memory,
                    "cpu_delta": end_cpu - start_cpu
                }
                
                for metric in self.profiler_config.metrics:
                    if metric in profile:
                        self._profiles[metric].append(profile)
                        
                        # Trim profiles if needed
                        if len(self._profiles[metric]) > self.profiler_config.max_samples:
                            self._profiles[metric] = self._profiles[metric][-self.profiler_config.max_samples:]
                            
                return result
                
            except Exception as e:
                # Record error metrics
                end_time = datetime.now()
                profile = {
                    "timestamp": end_time,
                    "operation": operation,
                    "error": str(e),
                    "duration": (end_time - start_time).total_seconds()
                }
                
                for metric in self.profiler_config.metrics:
                    if metric in profile:
                        self._profiles[metric].append(profile)
                        
                raise
                
        except Exception as e:
            raise DevToolsError(f"Failed to profile operation: {str(e)}")
            
    async def get_profiles(
        self,
        metric: str,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get profiling data.
        
        Args:
            metric: Metric to retrieve
            operation: Optional operation name to filter by
            start_time: Optional start time for profiles
            end_time: Optional end time for profiles
            
        Returns:
            List of profile data
            
        Raises:
            DevToolsError: If profile retrieval fails
        """
        try:
            if metric not in self._profiles:
                return []
                
            profiles = self._profiles[metric]
            
            # Filter by operation
            if operation:
                profiles = [p for p in profiles if p["operation"] == operation]
                
            # Filter by time range
            if start_time or end_time:
                profiles = [
                    p for p in profiles
                    if (not start_time or p["timestamp"] >= start_time) and
                       (not end_time or p["timestamp"] <= end_time)
                ]
                
            return profiles
            
        except Exception as e:
            raise DevToolsError(f"Failed to get profiles: {str(e)}")
            
    async def _get_memory_usage(self) -> float:
        """Get memory usage in bytes."""
        try:
            # Implementation depends on platform
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0
            
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            # Implementation depends on platform
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting CPU usage: {str(e)}")
            return 0.0
            
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            # Clear traces
            self._traces.clear()
            
            # Clear profiles
            self._profiles.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup() 