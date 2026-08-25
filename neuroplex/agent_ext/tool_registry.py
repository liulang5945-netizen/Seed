"""Process-wide tool registry.

Real implementation of the registry previously stubbed here. Tools are
declared via :class:`ToolDef` and stored in a name-keyed dict; ``execute``
dispatches keyword arguments to the handler. ``ToolDef`` accepts both the
canonical ``handler`` keyword and the legacy ``func`` alias used by
``neuroplex/tools/builtin_tools.py``.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolDef:
    """Declarative tool definition."""

    name: str = ""
    description: str = ""
    parameters: dict | None = None
    handler: Callable | None = None
    func: Callable | None = None  # legacy alias for handler
    category: str = ""
    source: str = "builtin"

    @property
    def callable(self) -> Callable | None:
        return self.handler if self.handler is not None else self.func


class ToolRegistry:
    """Name-keyed registry with dispatch."""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        if tool is None or not getattr(tool, "name", ""):
            logger.warning("Ignoring tool registration without a name")
            return
        self._tools[tool.name] = tool

    def list_tools(self) -> list[ToolDef]:
        return list(self._tools.values())

    def get_tool(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def execute(self, name: str, **kwargs):
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered")
        fn = tool.callable
        if fn is None:
            raise NotImplementedError(f"Tool '{name}' has no handler")
        return fn(**kwargs)


registry = ToolRegistry()
