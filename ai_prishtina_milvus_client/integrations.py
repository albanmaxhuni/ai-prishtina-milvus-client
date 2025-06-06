"""
Integration utilities for Milvus operations with async support.
"""

import asyncio
from typing import Dict, Any, Optional, List, Union
import aiohttp
import json
from datetime import datetime
from pydantic import BaseModel, Field
from .exceptions import IntegrationError

class IntegrationConfig(BaseModel):
    """Configuration for integrations."""
    enabled: bool = Field(True, description="Whether the integration is enabled")
    timeout: float = Field(30.0, description="Request timeout in seconds")
    retry_count: int = Field(3, description="Number of retries for failed requests")
    retry_delay: float = Field(1.0, description="Delay between retries in seconds")
    headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    verify_ssl: bool = Field(True, description="Whether to verify SSL certificates")

class SlackIntegration:
    """Integration with Slack for notifications."""
    
    def __init__(self, webhook_url: str, config: Optional[IntegrationConfig] = None):
        """
        Initialize Slack integration.
        
        Args:
            webhook_url: Slack webhook URL
            config: Optional integration configuration
        """
        self.webhook_url = webhook_url
        self.config = config or IntegrationConfig()
        
    async def send_message(
        self,
        message: str,
        channel: Optional[str] = None,
        username: Optional[str] = None,
        icon_emoji: Optional[str] = None
    ) -> bool:
        """
        Send a message to Slack asynchronously.
        
        Args:
            message: Message to send
            channel: Optional channel to send to
            username: Optional username to send as
            icon_emoji: Optional emoji to use as icon
            
        Returns:
            Whether the message was sent successfully
            
        Raises:
            IntegrationError: If sending fails
        """
        try:
            payload = {
                "text": message,
                "channel": channel,
                "username": username,
                "icon_emoji": icon_emoji
            }
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(self.config.retry_count):
                    try:
                        async with session.post(
                            self.webhook_url,
                            json=payload,
                            headers=self.config.headers,
                            timeout=self.config.timeout,
                            ssl=self.config.verify_ssl
                        ) as response:
                            if response.status == 200:
                                return True
                            raise IntegrationError(f"Slack API error: {response.status}")
                    except asyncio.TimeoutError:
                        if attempt == self.config.retry_count - 1:
                            raise IntegrationError("Slack request timed out")
                        await asyncio.sleep(self.config.retry_delay)
                    except Exception as e:
                        if attempt == self.config.retry_count - 1:
                            raise IntegrationError(f"Slack request failed: {str(e)}")
                        await asyncio.sleep(self.config.retry_delay)
                        
            return False
        except Exception as e:
            raise IntegrationError(f"Slack integration failed: {str(e)}")

class EmailIntegration:
    """Integration with email for notifications."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        config: Optional[IntegrationConfig] = None
    ):
        """
        Initialize email integration.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            config: Optional integration configuration
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.config = config or IntegrationConfig()
        
    async def send_email(
        self,
        to_addresses: Union[str, List[str]],
        subject: str,
        body: str,
        cc_addresses: Optional[Union[str, List[str]]] = None,
        bcc_addresses: Optional[Union[str, List[str]]] = None,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Send an email asynchronously.
        
        Args:
            to_addresses: Recipient email address(es)
            subject: Email subject
            body: Email body
            cc_addresses: Optional CC email address(es)
            bcc_addresses: Optional BCC email address(es)
            attachments: Optional list of attachments
            
        Returns:
            Whether the email was sent successfully
            
        Raises:
            IntegrationError: If sending fails
        """
        try:
            import aiosmtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from email.mime.application import MIMEApplication
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.username
            msg["To"] = to_addresses if isinstance(to_addresses, str) else ", ".join(to_addresses)
            if cc_addresses:
                msg["Cc"] = cc_addresses if isinstance(cc_addresses, str) else ", ".join(cc_addresses)
            if bcc_addresses:
                msg["Bcc"] = bcc_addresses if isinstance(bcc_addresses, str) else ", ".join(bcc_addresses)
            msg["Subject"] = subject
            
            # Add body
            msg.attach(MIMEText(body, "plain"))
            
            # Add attachments
            if attachments:
                for attachment in attachments:
                    part = MIMEApplication(attachment["data"])
                    part.add_header(
                        "Content-Disposition",
                        f"attachment; filename={attachment['filename']}"
                    )
                    msg.attach(part)
                    
            # Send email
            for attempt in range(self.config.retry_count):
                try:
                    async with aiosmtplib.SMTP(
                        hostname=self.smtp_host,
                        port=self.smtp_port,
                        use_tls=True,
                        timeout=self.config.timeout
                    ) as smtp:
                        await smtp.login(self.username, self.password)
                        await smtp.send_message(msg)
                        return True
                except asyncio.TimeoutError:
                    if attempt == self.config.retry_count - 1:
                        raise IntegrationError("SMTP request timed out")
                    await asyncio.sleep(self.config.retry_delay)
                except Exception as e:
                    if attempt == self.config.retry_count - 1:
                        raise IntegrationError(f"SMTP request failed: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay)
                    
            return False
        except Exception as e:
            raise IntegrationError(f"Email integration failed: {str(e)}")

class PrometheusIntegration:
    """Integration with Prometheus for metrics."""
    
    def __init__(self, push_gateway_url: str, config: Optional[IntegrationConfig] = None):
        """
        Initialize Prometheus integration.
        
        Args:
            push_gateway_url: Prometheus Pushgateway URL
            config: Optional integration configuration
        """
        self.push_gateway_url = push_gateway_url
        self.config = config or IntegrationConfig()
        
    async def push_metrics(
        self,
        metrics: Dict[str, float],
        job: str,
        instance: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Push metrics to Prometheus asynchronously.
        
        Args:
            metrics: Dictionary of metric names and values
            job: Job name
            instance: Optional instance name
            labels: Optional additional labels
            
        Returns:
            Whether the metrics were pushed successfully
            
        Raises:
            IntegrationError: If pushing fails
        """
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
            
            # Create registry and metrics
            registry = CollectorRegistry()
            for name, value in metrics.items():
                Gauge(name, f"Metric {name}", registry=registry).set(value)
                
            # Push metrics
            for attempt in range(self.config.retry_count):
                try:
                    await asyncio.to_thread(
                        push_to_gateway,
                        self.push_gateway_url,
                        job=job,
                        instance=instance,
                        registry=registry,
                        grouping_key=labels
                    )
                    return True
                except Exception as e:
                    if attempt == self.config.retry_count - 1:
                        raise IntegrationError(f"Prometheus push failed: {str(e)}")
                    await asyncio.sleep(self.config.retry_delay)
                    
            return False
        except Exception as e:
            raise IntegrationError(f"Prometheus integration failed: {str(e)}")

class GrafanaIntegration:
    """Integration with Grafana for dashboards."""
    
    def __init__(self, api_url: str, api_key: str, config: Optional[IntegrationConfig] = None):
        """
        Initialize Grafana integration.
        
        Args:
            api_url: Grafana API URL
            api_key: Grafana API key
            config: Optional integration configuration
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.config = config or IntegrationConfig()
        self.config.headers["Authorization"] = f"Bearer {api_key}"
        
    async def create_dashboard(
        self,
        title: str,
        panels: List[Dict[str, Any]],
        tags: Optional[List[str]] = None,
        folder_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Create a Grafana dashboard asynchronously.
        
        Args:
            title: Dashboard title
            panels: List of dashboard panels
            tags: Optional list of tags
            folder_id: Optional folder ID
            
        Returns:
            Created dashboard data
            
        Raises:
            IntegrationError: If creation fails
        """
        try:
            dashboard = {
                "dashboard": {
                    "title": title,
                    "panels": panels,
                    "tags": tags or [],
                    "timezone": "browser",
                    "schemaVersion": 30,
                    "version": 0,
                    "refresh": "5s"
                },
                "folderId": folder_id,
                "overwrite": False
            }
            
            async with aiohttp.ClientSession() as session:
                for attempt in range(self.config.retry_count):
                    try:
                        async with session.post(
                            f"{self.api_url}/api/dashboards/db",
                            json=dashboard,
                            headers=self.config.headers,
                            timeout=self.config.timeout,
                            ssl=self.config.verify_ssl
                        ) as response:
                            if response.status == 200:
                                return await response.json()
                            raise IntegrationError(f"Grafana API error: {response.status}")
                    except asyncio.TimeoutError:
                        if attempt == self.config.retry_count - 1:
                            raise IntegrationError("Grafana request timed out")
                        await asyncio.sleep(self.config.retry_delay)
                    except Exception as e:
                        if attempt == self.config.retry_count - 1:
                            raise IntegrationError(f"Grafana request failed: {str(e)}")
                        await asyncio.sleep(self.config.retry_delay)
                        
            raise IntegrationError("Failed to create dashboard after retries")
        except Exception as e:
            raise IntegrationError(f"Grafana integration failed: {str(e)}") 