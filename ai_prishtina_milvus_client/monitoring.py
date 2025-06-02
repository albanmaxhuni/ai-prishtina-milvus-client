"""
Monitoring and metrics collection for Milvus operations.
"""

from typing import Dict, Any, Optional, List, Callable
import time
from datetime import datetime
import threading
from collections import deque
import psutil
import numpy as np
from pydantic import BaseModel, Field

class PerformanceMetrics(BaseModel):
    """Performance metrics for operations."""
    operation_type: str = Field(..., description="Type of operation")
    start_time: datetime = Field(..., description="Operation start time")
    end_time: Optional[datetime] = Field(None, description="Operation end time")
    duration: Optional[float] = Field(None, description="Operation duration in seconds")
    items_processed: int = Field(0, description="Number of items processed")
    success_rate: float = Field(0.0, description="Operation success rate")
    error_count: int = Field(0, description="Number of errors encountered")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of errors")

class SystemMetrics(BaseModel):
    """System resource usage metrics."""
    timestamp: float = Field(..., description="Timestamp of metrics collection")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    disk_io_read: float = Field(..., description="Disk read bytes per second")
    disk_io_write: float = Field(..., description="Disk write bytes per second")
    network_io_sent: float = Field(..., description="Network sent bytes per second")
    network_io_recv: float = Field(..., description="Network received bytes per second")

class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""
    collect_system_metrics: bool = Field(True, description="Whether to collect system metrics")
    metrics_history_size: int = Field(1000, description="Number of metrics to keep in history")
    collection_interval: float = Field(1.0, description="System metrics collection interval in seconds")
    enable_alerting: bool = Field(False, description="Whether to enable alerting")
    alert_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "cpu_percent": 80.0,
            "memory_percent": 80.0,
            "disk_io_read": 1000000.0,  # 1MB/s
            "disk_io_write": 1000000.0,  # 1MB/s
        },
        description="Alert thresholds for system metrics"
    )
    metrics_retention: int = Field(3600, description="Metrics retention in seconds")
    alert_threshold: float = Field(0.8, description="General alert threshold")
    enable_logging: bool = Field(True, description="Enable logging for monitoring")

class MetricsCollector:
    """Collects and manages performance and system metrics."""
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize metrics collector.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.performance_metrics: deque = deque(maxlen=config.metrics_history_size)
        self.system_metrics: deque = deque(maxlen=config.metrics_history_size)
        self._stop_collection = threading.Event()
        self._collection_thread: Optional[threading.Thread] = None
        
        if config.collect_system_metrics:
            self._start_system_metrics_collection()
            
    def _start_system_metrics_collection(self):
        """Start system metrics collection in a background thread."""
        self._collection_thread = threading.Thread(
            target=self._collect_system_metrics,
            daemon=True
        )
        self._collection_thread.start()
        
    def _collect_system_metrics(self):
        """Collect system metrics periodically."""
        last_io = psutil.disk_io_counters()
        last_net = psutil.net_io_counters()
        last_time = time.time()
        
        while not self._stop_collection.is_set():
            try:
                # Get current metrics
                current_time = time.time()
                current_io = psutil.disk_io_counters()
                current_net = psutil.net_io_counters()
                
                # Calculate rates
                time_diff = current_time - last_time
                disk_read_rate = (current_io.read_bytes - last_io.read_bytes) / time_diff
                disk_write_rate = (current_io.write_bytes - last_io.write_bytes) / time_diff
                net_sent_rate = (current_net.bytes_sent - last_net.bytes_sent) / time_diff
                net_recv_rate = (current_net.bytes_recv - last_net.bytes_recv) / time_diff
                
                # Create metrics
                metrics = SystemMetrics(
                    timestamp=current_time,
                    cpu_percent=psutil.cpu_percent(),
                    memory_percent=psutil.virtual_memory().percent,
                    disk_io_read=disk_read_rate,
                    disk_io_write=disk_write_rate,
                    network_io_sent=net_sent_rate,
                    network_io_recv=net_recv_rate
                )
                
                # Store metrics
                self.system_metrics.append(metrics)
                
                # Check alert thresholds
                if self.config.enable_alerting:
                    self._check_alert_thresholds(metrics)
                    
                # Update last values
                last_io = current_io
                last_net = current_net
                last_time = current_time
                
                # Wait for next collection
                time.sleep(self.config.collection_interval)
                
            except Exception as e:
                print(f"Error collecting system metrics: {str(e)}")
                time.sleep(self.config.collection_interval)
                
    def _check_alert_thresholds(self, metrics: SystemMetrics):
        """
        Check if any metrics exceed alert thresholds.
        
        Args:
            metrics: System metrics to check
        """
        for metric, threshold in self.config.alert_thresholds.items():
            value = getattr(metrics, metric)
            if value > threshold:
                self._trigger_alert(metric, value, threshold)
                
    def _trigger_alert(self, metric: str, value: float, threshold: float):
        """
        Trigger an alert for exceeded threshold.
        
        Args:
            metric: Metric name
            value: Current value
            threshold: Threshold value
        """
        print(f"ALERT: {metric} exceeded threshold. Current: {value}, Threshold: {threshold}")
        # TODO: Implement proper alerting (e.g., email, Slack, etc.)
        
    def record_operation(
        self,
        operation_type: str,
        items_processed: int,
        error_count: int,
        errors: List[Dict[str, Any]]
    ) -> PerformanceMetrics:
        """
        Record performance metrics for an operation.
        
        Args:
            operation_type: Type of operation
            items_processed: Number of items processed
            error_count: Number of errors encountered
            errors: List of error details
            
        Returns:
            Performance metrics
        """
        metrics = PerformanceMetrics(
            operation_type=operation_type,
            start_time=datetime.now(),
            items_processed=items_processed,
            error_count=error_count,
            errors=errors
        )
        
        self.performance_metrics.append(metrics)
        return metrics
        
    def complete_operation(self, metrics: PerformanceMetrics):
        """
        Complete a performance metrics record.
        
        Args:
            metrics: Performance metrics to complete
        """
        metrics.end_time = datetime.now()
        metrics.duration = (metrics.end_time - metrics.start_time).total_seconds()
        metrics.success_rate = (
            (metrics.items_processed - metrics.error_count) / metrics.items_processed
            if metrics.items_processed > 0
            else 0.0
        )
        
    def get_performance_summary(
        self,
        operation_type: Optional[str] = None,
        time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get summary of performance metrics.
        
        Args:
            operation_type: Optional operation type to filter by
            time_window: Optional time window in seconds to consider
            
        Returns:
            Dictionary with performance summary
        """
        # Filter metrics
        metrics = self.performance_metrics
        if operation_type:
            metrics = [m for m in metrics if m.operation_type == operation_type]
        if time_window:
            cutoff = datetime.now().timestamp() - time_window
            metrics = [m for m in metrics if m.start_time.timestamp() > cutoff]
            
        if not metrics:
            return {
                "total_operations": 0,
                "total_items": 0,
                "average_duration": 0.0,
                "average_success_rate": 0.0,
                "total_errors": 0
            }
            
        # Calculate summary
        total_operations = len(metrics)
        total_items = sum(m.items_processed for m in metrics)
        total_errors = sum(m.error_count for m in metrics)
        durations = [m.duration for m in metrics if m.duration is not None]
        success_rates = [m.success_rate for m in metrics]
        
        return {
            "total_operations": total_operations,
            "total_items": total_items,
            "average_duration": np.mean(durations) if durations else 0.0,
            "average_success_rate": np.mean(success_rates) if success_rates else 0.0,
            "total_errors": total_errors
        }
        
    def get_system_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """
        Get summary of system metrics.
        
        Args:
            time_window: Optional time window in seconds to consider
            
        Returns:
            Dictionary with system metrics summary
        """
        # Filter metrics
        metrics = self.system_metrics
        if time_window:
            cutoff = datetime.now().timestamp() - time_window
            metrics = [m for m in metrics if m.timestamp > cutoff]
            
        if not metrics:
            return {
                "average_cpu": 0.0,
                "average_memory": 0.0,
                "average_disk_io": 0.0,
                "average_network_io": 0.0
            }
            
        # Calculate summary
        return {
            "average_cpu": np.mean([m.cpu_percent for m in metrics]),
            "average_memory": np.mean([m.memory_percent for m in metrics]),
            "average_disk_io": np.mean([
                m.disk_io_read + m.disk_io_write for m in metrics
            ]),
            "average_network_io": np.mean([
                m.network_io_sent + m.network_io_recv for m in metrics
            ])
        }
        
    def stop(self):
        """Stop metrics collection."""
        self._stop_collection.set()
        if self._collection_thread:
            self._collection_thread.join() 