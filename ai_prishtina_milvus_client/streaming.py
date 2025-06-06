"""
Streaming support for real-time vector ingestion and search with async support.
"""

import json
from typing import Any, Dict, List, Optional, Union, Awaitable
from dataclasses import dataclass
import asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel, Field

from .client import AsyncMilvusClient
from .config import MilvusConfig


class StreamConfig(BaseModel):
    """Configuration for streaming."""
    bootstrap_servers: str = Field(..., description="Kafka bootstrap servers")
    group_id: str = Field(..., description="Consumer group ID")
    topics: List[str] = Field(..., description="Kafka topics to consume from")
    auto_offset_reset: str = Field("earliest", description="Auto offset reset policy")
    enable_auto_commit: bool = Field(True, description="Enable auto commit")
    max_poll_interval_ms: int = Field(300000, description="Max poll interval in ms")
    session_timeout_ms: int = Field(10000, description="Session timeout in ms")
    max_poll_records: int = Field(500, description="Max poll records")
    batch_size: int = Field(1000, description="Batch size for vector insertion")
    num_workers: int = Field(4, description="Number of worker tasks")


@dataclass
class StreamMessage:
    """Stream message format."""
    vectors: List[List[float]]
    metadata: Optional[List[Dict[str, Any]]] = None
    operation: str = "insert"  # insert, delete, update
    collection: str = "default"


class KafkaStreamProcessor:
    """Kafka stream processor for real-time vector ingestion."""
    
    def __init__(
        self,
        milvus_config: MilvusConfig,
        stream_config: StreamConfig,
        client: Optional[AsyncMilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.stream_config = stream_config
        self.client = client or AsyncMilvusClient(milvus_config)
        self.consumer = None
        self.producer = None
        self.batch_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.workers = []
        
    async def initialize(self):
        """Initialize Kafka consumer and producer asynchronously."""
        self.consumer = AIOKafkaConsumer(
            *self.stream_config.topics,
            bootstrap_servers=self.stream_config.bootstrap_servers,
            group_id=self.stream_config.group_id,
            auto_offset_reset=self.stream_config.auto_offset_reset,
            enable_auto_commit=self.stream_config.enable_auto_commit,
            max_poll_interval_ms=self.stream_config.max_poll_interval_ms,
            session_timeout_ms=self.stream_config.session_timeout_ms,
            max_poll_records=self.stream_config.max_poll_records
        )
        
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.stream_config.bootstrap_servers
        )
        
        await self.consumer.start()
        await self.producer.start()
        
    async def start(self):
        """Start processing messages asynchronously."""
        if not self.consumer or not self.producer:
            await self.initialize()
            
        # Start worker tasks
        for _ in range(self.stream_config.num_workers):
            worker = asyncio.create_task(self._process_batches())
            self.workers.append(worker)
            
        # Start consuming messages
        try:
            while not self.stop_event.is_set():
                msg = await self.consumer.getone()
                if msg is None:
                    continue
                    
                # Parse message
                try:
                    data = json.loads(msg.value.decode("utf-8"))
                    message = StreamMessage(**data)
                    await self.batch_queue.put(message)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
                    
        except asyncio.CancelledError:
            await self.stop()
            
    async def stop(self):
        """Stop processing messages asynchronously."""
        self.stop_event.set()
        
        # Cancel worker tasks
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to complete
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Close Kafka connections
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()
        await self.client.close()
        
    async def _process_batches(self):
        """Process batches of messages asynchronously."""
        batch = []
        while not self.stop_event.is_set():
            try:
                message = await asyncio.wait_for(
                    self.batch_queue.get(),
                    timeout=1.0
                )
                batch.append(message)
                
                if len(batch) >= self.stream_config.batch_size:
                    await self._insert_batch(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self._insert_batch(batch)
                    batch = []
                    
    async def _insert_batch(self, batch: List[StreamMessage]):
        """Insert a batch of messages asynchronously."""
        try:
            vectors = []
            metadata = []
            for msg in batch:
                vectors.extend(msg.vectors)
                if msg.metadata:
                    metadata.extend(msg.metadata)
                    
            if vectors:
                await self.client.insert(vectors, metadata)
        except Exception as e:
            print(f"Error inserting batch: {e}")
            
    async def produce_message(self, topic: str, message: StreamMessage):
        """Produce a message to Kafka asynchronously."""
        try:
            await self.producer.send_and_wait(
                topic,
                json.dumps(message.__dict__).encode("utf-8")
            )
        except Exception as e:
            print(f"Error producing message: {e}")
            
    async def _delivery_report(self, err, msg):
        """Delivery report callback asynchronously."""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]") 