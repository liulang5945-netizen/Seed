"""Memory manager stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will back the agent's short-term / long-term memory
stores with persistent storage.
"""

import logging

logger = logging.getLogger(__name__)


class _MemoryStub:
    """Stub memory manager. All operations are no-ops returning safe defaults."""

    def get_status(self):
        return {"enabled": False, "message": "memory_manager not implemented"}

    def get_context(self, last_n=20):
        return []

    def add_message(self, role, content):
        pass

    def clear(self):
        pass

    # R5: 补齐 routes_agent_memory 调用的接口面，消除 mypy attr-defined
    # （此前缺失时运行时靠路由 try/except 吞 AttributeError）。
    def remember(self, text, category="general"):
        pass

    def recall(self, query, top_k=5):
        return []

    def set_working(self, key, value):
        pass

    def clear_all(self):
        pass

    class _WorkingStub:
        def get_all(self):
            return {}

        def list_keys(self):
            return []

        def clear(self):
            pass

    class _LongTermStub:
        def list_entries(self, category=None, limit=50):
            return []

        def count(self):
            return 0

        def clear(self):
            pass

    class _ShortTermStub:
        def clear(self):
            pass

    working = _WorkingStub()
    long_term = _LongTermStub()
    short_term = _ShortTermStub()


memory = _MemoryStub()
