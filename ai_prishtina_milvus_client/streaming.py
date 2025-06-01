"""
Streaming support for real-time vector ingestion and search.
"""

import json
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue

from confluent_kafka import Consumer, Producer, KafkaError
from pydantic import BaseModel, Field

from .client import MilvusClient
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
    num_workers: int = Field(4, description="Number of worker threads")


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
        client: Optional[MilvusClient] = None
    ):
        self.milvus_config = milvus_config
        self.stream_config = stream_config
        self.client = client or MilvusClient(milvus_config)
        self.consumer = Consumer({
            "bootstrap.servers": stream_config.bootstrap_servers,
            "group.id": stream_config.group_id,
            "auto.offset.reset": stream_config.auto_offset_reset,
            "enable.auto.commit": stream_config.enable_auto_commit,
            "max.poll.interval.ms": stream_config.max_poll_interval_ms,
            "session.timeout.ms": stream_config.session_timeout_ms,
            "max.poll.records": stream_config.max_poll_records
        })
        self.producer = Producer({
            "bootstrap.servers": stream_config.bootstrap_servers
        })
        self.consumer.subscribe(stream_config.topics)
        self.batch_queue = Queue()
        self.stop_event = threading.Event()
        self.workers = []
        
    def start(self):
        """Start processing messages."""
        # Start worker threads
        for _ in range(self.stream_config.num_workers):
            worker = threading.Thread(target=self._process_batches)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
        # Start consuming messages
        try:
            while not self.stop_event.is_set():
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    raise Exception(f"Consumer error: {msg.error()}")
                    
                # Parse message
                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    message = StreamMessage(**data)
                    self.batch_queue.put(message)
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            self.stop()
            
    def stop(self):
        """Stop processing messages."""
        self.stop_event.set()
        for worker in self.workers:
            worker.join()
        self.consumer.close()
        self.client.close()
        
    def _process_batches(self):
        """Process batches of messages."""
        batch = []
        while not self.stop_event.is_set():
            try:
                message = self.batch_queue.get(timeout=1.0)
                batch.append(message)
                
                if len(batch) >= self.stream_config.batch_size:
                    self._insert_batch(batch)
                    batch = []
            except Queue.Empty:
                if batch:
                    self._insert_batch(batch)
                    batch = []
                    
    def _insert_batch(self, batch: List[StreamMessage]):
        """Insert a batch of messages."""
        try:
            vectors = []
            metadata = []
            for msg in batch:
                vectors.extend(msg.vectors)
                if msg.metadata:
                    metadata.extend(msg.metadata)
                    
            if vectors:
                self.client.insert(vectors, metadata)
        except Exception as e:
            print(f"Error inserting batch: {e}")
            
    def produce_message(self, topic: str, message: StreamMessage):
        """Produce a message to Kafka."""
        try:
            self.producer.produce(
                topic,
                json.dumps(message.__dict__).encode("utf-8"),
                callback=self._delivery_report
            )
            self.producer.poll(0)
        except Exception as e:
            print(f"Error producing message: {e}")
            
    def _delivery_report(self, err, msg):
        """Delivery report callback."""
        if err is not None:
            print(f"Message delivery failed: {err}")
        else:
            print(f"Message delivered to {msg.topic()} [{msg.partition()}]") 