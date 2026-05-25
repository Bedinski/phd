"""In-process MCP servers exposed to agent stages."""

from .phd_mcp import build_phd_mcp_server

__all__ = ["build_phd_mcp_server"]
