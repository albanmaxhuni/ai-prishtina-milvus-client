"""
Monitoring and metrics collection with async support.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, Awaitable
from datetime import datetime, timedelta
import json

from pydantic import BaseModel, Field

from .exceptions import MonitoringError
from .client import AsyncMilvusClient
from .config import MilvusConfig


class MonitoringConfig(BaseModel):
    """Configuration for monitoring."""
    collect_system_metrics: bool = Field(True, description="Whether to collect system metrics")
    metrics_history_size: int = Field(1000, description="Number of metrics points to keep in history")
    collection_interval: float = Field(1.0, description="Interval between metric collections in seconds")


class PerformanceMetrics(BaseModel):
    """Performance metrics data model."""
    query_latency: float = Field(0.0, description="Average query latency in milliseconds")
    query_count: int = Field(0, description="Total number of queries")
    error_count: int = Field(0, description="Total number of errors")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the metrics")


class SystemMetrics(BaseModel):
    """System metrics data model."""
    cpu_usage: float = Field(0.0, description="CPU usage percentage")
    memory_usage: float = Field(0.0, description="Memory usage percentage")
    disk_usage: float = Field(0.0, description="Disk usage percentage")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the metrics")


class MetricsCollector:
    """Collects and manages metrics with async support."""
    
    def __init__(self, config: Optional[MonitoringConfig] = None):
        self.config = config or MonitoringConfig()
        self.logger = logging.getLogger(__name__)
        self._performance_metrics: List[PerformanceMetrics] = []
        self._system_metrics: List[SystemMetrics] = []
        self._collection_task: Optional[asyncio.Task] = None
        self._is_running = False
        
    async def start(self) -> None:
        """Start metrics collection."""
        if self._is_running:
            return
            
        self._is_running = True
        self._collection_task = asyncio.create_task(self._collect_metrics())
        
    async def stop(self) -> None:
        """Stop metrics collection."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
            self._collection_task = None
            
    async def get_performance_metrics(self) -> List[PerformanceMetrics]:
        """Get collected performance metrics."""
        return self._performance_metrics[-self.config.metrics_history_size:]
        
    async def get_system_metrics(self) -> List[SystemMetrics]:
        """Get collected system metrics."""
        return self._system_metrics[-self.config.metrics_history_size:]
        
    async def _collect_metrics(self) -> None:
        """Collect metrics periodically."""
        while self._is_running:
            try:
                # Collect performance metrics
                perf_metrics = await self._collect_performance_metrics()
                self._performance_metrics.append(perf_metrics)
                
                # Collect system metrics if enabled
                if self.config.collect_system_metrics:
                    sys_metrics = await self._collect_system_metrics()
                    self._system_metrics.append(sys_metrics)
                    
                # Cleanup old metrics
                self._cleanup_old_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(self.config.collection_interval)
                
    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect performance metrics."""
        return PerformanceMetrics(
            query_latency=await self._get_average_latency(),
            query_count=await self._get_query_count(),
            error_count=await self._get_error_count()
        )
        
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system metrics."""
        return SystemMetrics(
            cpu_usage=await self._get_cpu_usage(),
            memory_usage=await self._get_memory_usage(),
            disk_usage=await self._get_disk_usage()
        )
        
    def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond the history size limit."""
        if len(self._performance_metrics) > self.config.metrics_history_size:
            self._performance_metrics = self._performance_metrics[-self.config.metrics_history_size:]
        if len(self._system_metrics) > self.config.metrics_history_size:
            self._system_metrics = self._system_metrics[-self.config.metrics_history_size:]


class MetricConfig(BaseModel):
    """Configuration for metrics collection."""
    collection_interval: float = Field(60.0, description="Metrics collection interval in seconds")
    retention_period: float = Field(86400.0, description="Metrics retention period in seconds")
    enabled_metrics: List[str] = Field(
        default=["performance", "resource_usage", "query_stats"],
        description="List of enabled metrics"
    )


class AlertConfig(BaseModel):
    """Configuration for alerting."""
    enabled: bool = Field(True, description="Whether alerting is enabled")
    thresholds: Dict[str, float] = Field(
        default={
            "cpu_usage": 80.0,
            "memory_usage": 80.0,
            "query_latency": 1000.0,
            "error_rate": 5.0
        },
        description="Alert thresholds for different metrics"
    )
    notification_channels: List[str] = Field(
        default=["log"],
        description="Notification channels for alerts"
    )


class MonitoringManager:
    """Manager for monitoring and metrics collection with async support."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        metric_config: Optional[MetricConfig] = None,
        alert_config: Optional[AlertConfig] = None,
        client: Optional[AsyncMilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.metric_config = metric_config or MetricConfig()
        self.alert_config = alert_config or AlertConfig()
        self.client = client or AsyncMilvusClient(milvus_config)
        self.logger = logging.getLogger(__name__)
        self._metrics: Dict[str, List[Dict[str, Any]]] = {}
        self._collection_task: Optional[asyncio.Task] = None
        self._is_running = False
        
    async def start(self) -> None:
        """Start metrics collection."""
        try:
            if self._is_running:
                return
                
            self._is_running = True
            self._collection_task = asyncio.create_task(self._collect_metrics())
            
        except Exception as e:
            raise MonitoringError(f"Failed to start monitoring: {str(e)}")
            
    async def stop(self) -> None:
        """Stop metrics collection."""
        try:
            if not self._is_running:
                return
                
            self._is_running = False
            if self._collection_task:
                self._collection_task.cancel()
                try:
                    await self._collection_task
                except asyncio.CancelledError:
                    pass
                self._collection_task = None
                
        except Exception as e:
            raise MonitoringError(f"Failed to stop monitoring: {str(e)}")
            
    async def get_metrics(
        self,
        metric_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get collected metrics.
        
        Args:
            metric_type: Type of metrics to retrieve
            start_time: Start time for metrics range
            end_time: End time for metrics range
            
        Returns:
            List of metric data points
            
        Raises:
            MonitoringError: If metrics retrieval fails
        """
        try:
            if metric_type not in self._metrics:
                return []
                
            metrics = self._metrics[metric_type]
            
            # Filter by time range
            if start_time or end_time:
                metrics = [
                    m for m in metrics
                    if (not start_time or m["timestamp"] >= start_time) and
                       (not end_time or m["timestamp"] <= end_time)
                ]
                
            return metrics
            
        except Exception as e:
            raise MonitoringError(f"Failed to get metrics: {str(e)}")
            
    async def _collect_metrics(self) -> None:
        """Collect metrics periodically."""
        while self._is_running:
            try:
                # Collect performance metrics
                if "performance" in self.metric_config.enabled_metrics:
                    await self._collect_performance_metrics()
                    
                # Collect resource usage metrics
                if "resource_usage" in self.metric_config.enabled_metrics:
                    await self._collect_resource_metrics()
                    
                # Collect query statistics
                if "query_stats" in self.metric_config.enabled_metrics:
                    await self._collect_query_metrics()
                    
                # Check for alerts
                if self.alert_config.enabled:
                    await self._check_alerts()
                    
                # Cleanup old metrics
                await self._cleanup_old_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.metric_config.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {str(e)}")
                await asyncio.sleep(self.metric_config.collection_interval)
                
    async def _collect_performance_metrics(self) -> None:
        """Collect performance metrics."""
        try:
            metrics = {
                "timestamp": datetime.now(),
                "collection_stats": await self.client.get_collection_stats(),
                "index_stats": await self.client.get_index_stats(),
                "query_latency": await self._measure_query_latency()
            }
            
            self._metrics.setdefault("performance", []).append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {str(e)}")
            
    async def _collect_resource_metrics(self) -> None:
        """Collect resource usage metrics."""
        try:
            metrics = {
                "timestamp": datetime.now(),
                "cpu_usage": await self._get_cpu_usage(),
                "memory_usage": await self._get_memory_usage(),
                "disk_usage": await self._get_disk_usage()
            }
            
            self._metrics.setdefault("resource_usage", []).append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting resource metrics: {str(e)}")
            
    async def _collect_query_metrics(self) -> None:
        """Collect query statistics."""
        try:
            metrics = {
                "timestamp": datetime.now(),
                "query_count": await self._get_query_count(),
                "error_count": await self._get_error_count(),
                "average_latency": await self._get_average_latency()
            }
            
            self._metrics.setdefault("query_stats", []).append(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting query metrics: {str(e)}")
            
    async def _check_alerts(self) -> None:
        """Check for alert conditions."""
        try:
            # Get latest metrics
            for metric_type in self._metrics:
                if not self._metrics[metric_type]:
                    continue
                    
                latest = self._metrics[metric_type][-1]
                
                # Check thresholds
                for metric, value in latest.items():
                    if metric in self.alert_config.thresholds:
                        threshold = self.alert_config.thresholds[metric]
                        if value > threshold:
                            await self._send_alert(metric, value, threshold)
                            
        except Exception as e:
            self.logger.error(f"Error checking alerts: {str(e)}")
            
    async def _send_alert(self, metric: str, value: float, threshold: float) -> None:
        """Send alert notification."""
        try:
            message = f"Alert: {metric} exceeded threshold ({value} > {threshold})"
            
            for channel in self.alert_config.notification_channels:
                if channel == "log":
                    self.logger.warning(message)
                # Add more notification channels as needed
                
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")
            
    async def _cleanup_old_metrics(self) -> None:
        """Clean up old metrics data."""
        try:
            cutoff_time = datetime.now() - timedelta(seconds=self.metric_config.retention_period)
            
            for metric_type in self._metrics:
                self._metrics[metric_type] = [
                    m for m in self._metrics[metric_type]
                    if m["timestamp"] > cutoff_time
                ]
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old metrics: {str(e)}")
            
    async def _measure_query_latency(self) -> float:
        """Measure query latency."""
        try:
            start_time = datetime.now()
            await self.client.search_vectors([[0.0] * self.milvus_config.dim], top_k=1)
            return (datetime.now() - start_time).total_seconds() * 1000
            
        except Exception as e:
            self.logger.error(f"Error measuring query latency: {str(e)}")
            return 0.0
            
    async def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage."""
        try:
            # Implementation depends on platform
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting CPU usage: {str(e)}")
            return 0.0
            
    async def _get_memory_usage(self) -> float:
        """Get memory usage percentage."""
        try:
            # Implementation depends on platform
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting memory usage: {str(e)}")
            return 0.0
            
    async def _get_disk_usage(self) -> float:
        """Get disk usage percentage."""
        try:
            # Implementation depends on platform
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting disk usage: {str(e)}")
            return 0.0
            
    async def _get_query_count(self) -> int:
        """Get total query count."""
        try:
            # Implementation depends on metrics storage
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting query count: {str(e)}")
            return 0
            
    async def _get_error_count(self) -> int:
        """Get total error count."""
        try:
            # Implementation depends on metrics storage
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting error count: {str(e)}")
            return 0
            
    async def _get_average_latency(self) -> float:
        """Get average query latency."""
        try:
            # Implementation depends on metrics storage
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting average latency: {str(e)}")
            return 0.0
            
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.stop()
            self._metrics.clear()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            
    async def __aenter__(self):
        """Context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.cleanup() 