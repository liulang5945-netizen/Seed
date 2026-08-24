"""Tool service — bridges the agent_ext tool registry to the API layer.

Real implementation of the service previously stubbed here: exposes
registry entries as JSON-safe dicts (shape matches ``api.models_runtime.
ToolInfo``), renders JSON-Schema listings, and dispatches execution.
"""

import logging

from neuroplex.agent_ext.tool_registry import registry

logger = logging.getLogger(__name__)


def list_tools():
    """JSON-safe tool listing for /api/agent/tools and runtime status."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {},
            "source": tool.source or "builtin",
            "source_id": tool.name,
            "category": tool.category or "通用",
            "enabled": tool.callable is not None,
        }
        for tool in registry.list_tools()
    ]


def get_registry_schemas():
    """JSON-Schema style listing for /api/agent/tools/registry."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.parameters or {},
            "category": tool.category or "通用",
        }
        for tool in registry.list_tools()
    ]


def execute_tool(name, args):
    """Dispatch execution to the registered handler."""
    return registry.execute(name, **(args or {}))
