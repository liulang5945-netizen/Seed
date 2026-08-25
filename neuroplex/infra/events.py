"""进程内事件总线（EventBus）。

消费方约定（既有代码依赖的 API 形状）：

- ``get_event_bus()`` 返回进程级单例；
- ``publish(event_type, data, source=...)`` 发布事件，消息形状为
  ``{"event_type", "data", "source", "timestamp"}``；
- ``subscribe(event_type, handler)`` 订阅；``"*"`` 订阅所有事件；
- ``set_broadcast_callback(cb)`` 注册唯一广播回调（WebSocket 服务器用它
  把生命事件实时推给前端），回调接收完整消息字典。

发布侧异常隔离：任何订阅者抛错只记日志，不影响发布者与其他订阅者。
"""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable

logger = logging.getLogger(__name__)

WILDCARD = "*"


class EventBus:
    """线程安全的同步事件总线。"""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = defaultdict(list)
        self._broadcast_callback: Callable | None = None
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, handler: Callable) -> None:
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)

    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        with self._lock:
            handlers = self._subscribers.get(event_type) or []
            if handler in handlers:
                handlers.remove(handler)

    def set_broadcast_callback(self, callback: Callable) -> None:
        with self._lock:
            self._broadcast_callback = callback

    def publish(self, event_type: str, data: dict | None = None, source: str = "") -> dict:
        message = {
            "event_type": event_type,
            "data": data or {},
            "source": source,
            "timestamp": time.time(),
        }
        with self._lock:
            handlers = list(self._subscribers.get(event_type, []))
            handlers.extend(self._subscribers.get(WILDCARD, []))
            broadcast = self._broadcast_callback
        for handler in handlers:
            try:
                handler(message)
            except Exception as exc:
                logger.warning(f"EventBus subscriber failed for '{event_type}': {exc}")
        if broadcast is not None:
            try:
                broadcast(message)
            except Exception as exc:
                logger.warning(f"EventBus broadcast callback failed: {exc}")
        return message


_bus = EventBus()


def get_event_bus() -> EventBus:
    """进程级单例。"""
    return _bus
