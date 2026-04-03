"""Async Event Bus — Pub/Sub architecture for decoupling UI, Scanner and Alerts."""

import asyncio
from typing import Callable, Coroutine
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Asynchronous Event Bus using asyncio.Queue for multi-producer, multi-consumer streaming."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_bus()
        return cls._instance
        
    def _init_bus(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._queue = asyncio.Queue()
        self._worker_task = None
        
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a specific topic."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)
        
    async def publish(self, topic: str, data: dict):
        """Publish an event payload to the queue."""
        await self._queue.put({"topic": topic, "data": data})
        
    def publish_sync(self, topic: str, data: dict):
        """Publish an event payload synchronously (e.g., from Streamlit)."""
        import asyncio
        for cb in self._subscribers.get(topic, []):
            if asyncio.iscoroutinefunction(cb):
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(cb(topic, data))
                    else:
                        loop.run_until_complete(cb(topic, data))
                except RuntimeError:
                    asyncio.run(cb(topic, data))
            else:
                cb(topic, data)
        
    def start_worker(self):
        """Start the background worker to process queue if not already running."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._process_queue())
            
    async def _process_queue(self):
        """Continuously route messages from queue to subscribers."""
        while True:
            try:
                event = await self._queue.get()
                topic = event["topic"]
                data = event["data"]
                
                for cb in self._subscribers.get(topic, []):
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.create_task(cb(topic, data))
                    else:
                        cb(topic, data)
                        
                self._queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing event {event}: {e}")
