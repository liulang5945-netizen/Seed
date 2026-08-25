"""MCP manager stub — not yet implemented.

This is a minimal stub that allows imports to succeed. The real
implementation will register / discover Model Context Protocol
servers and expose their tools to the agent runtime.
"""

import logging

logger = logging.getLogger(__name__)


class _MCPManagerStub:
    """Stub MCP manager. Listings return empty, mutations raise."""

    def list_servers(self):
        return []

    def get_server(self, name):
        return None

    def add_server(self, *args, **kwargs):
        raise NotImplementedError

    def remove_server(self, name):
        raise NotImplementedError

    def list_tools(self, server_name=None):
        return []

    # R5: 补齐 routes_agent_mcp 调用的接口面，消除 mypy attr-defined。
    # 语义沿用存根策略：读返回空，写抛 NotImplementedError（由路由 try/except 兜底）。
    def get_marketplace(self, category="", keyword=""):
        return {"servers": [], "categories": []}

    def refresh_marketplace(self):
        raise NotImplementedError

    def get_server_detail(self, server_id):
        return None

    def install_server(self, server_id):
        raise NotImplementedError

    def uninstall_server(self, server_id):
        raise NotImplementedError

    def start_server(self, server_id, workspace_path=None):
        raise NotImplementedError

    def stop_server(self, server_id):
        raise NotImplementedError

    def restart_server(self, server_id, workspace_path=None):
        raise NotImplementedError

    def get_installed_servers(self):
        return []

    def get_status(self):
        return {"enabled": False, "message": "mcp_manager not implemented"}

    def get_all_mcp_tools(self):
        return []

    def get_plugin_marketplace(self, category="", keyword=""):
        return {"plugins": [], "categories": []}

    def refresh_plugin_marketplace(self):
        raise NotImplementedError

    def add_custom_server(self, *args, **kwargs):
        raise NotImplementedError


mcp_manager = _MCPManagerStub()
