"""Agent service stub — not yet implemented.

This is a minimal stub that allows imports to succeed as
``from neuroplex.services import agent_service``. The real implementation
will coordinate agent lifecycle, workspace, and task dispatch.
"""

import logging

logger = logging.getLogger(__name__)


# R5: 补齐 routes_agent 调用的接口面，消除 mypy attr-defined。
# 语义沿用存根策略：读返回空，写/执行抛 NotImplementedError（由路由 try/except 兜底）。


def run_react_task(task: str, max_steps: int = 15) -> dict:
    raise NotImplementedError


def run_react_stream(task: str, max_steps: int = 15):
    raise NotImplementedError


def cancel_active_task() -> str:
    return "no active task"


def list_roles() -> list:
    return []


def collaborate(task: str) -> dict:
    raise NotImplementedError


def list_collab_tasks() -> list:
    return []


def get_collab_messages(topic: str = "", limit: int = 50) -> list:
    return []
