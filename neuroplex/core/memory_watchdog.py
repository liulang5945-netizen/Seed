"""Compatibility exports for the platform-owned memory service."""

from seed_platform.memory import (
    MemoryWatchdog,
    force_memory_refresh,
    get_memory_status_dict,
    memory_guarded,
)

__all__ = [
    "MemoryWatchdog",
    "force_memory_refresh",
    "get_memory_status_dict",
    "memory_guarded",
]
