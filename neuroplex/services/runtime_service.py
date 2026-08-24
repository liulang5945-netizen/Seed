"""Compatibility exports for the platform-owned runtime status service."""

from seed_platform.runtime_service import get_bootstrap_status, get_runtime_status

__all__ = ["get_bootstrap_status", "get_runtime_status"]
