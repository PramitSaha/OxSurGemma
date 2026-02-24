"""
Tool registry for the surgical co-pilot.
Register tools by name; the agent loads only the tools you request.
Add new tools by implementing a BaseTool and registering them here.
"""
from typing import Callable, Dict, List, Optional

from langchain_core.tools import BaseTool

# Map: tool_name -> factory (callable that returns BaseTool)
_TOOL_REGISTRY: Dict[str, Callable[[], BaseTool]] = {}


def register(name: str):
    """Decorator to register a tool factory under a given name."""

    def decorator(factory: Callable[[], BaseTool]):
        _TOOL_REGISTRY[name] = factory
        return factory

    return decorator


def get_registered_names() -> List[str]:
    """Return all registered tool names."""
    return list(_TOOL_REGISTRY.keys())


def get_tools(
    tools_to_use: Optional[List[str]] = None,
) -> List[BaseTool]:
    """
    Return a list of tool instances.

    Args:
        tools_to_use: If None, all registered tools are instantiated.
                     Otherwise only tools whose name is in this list.

    Returns:
        List of BaseTool instances.
    """
    names = tools_to_use if tools_to_use is not None else get_registered_names()
    out = []
    for name in names:
        if name not in _TOOL_REGISTRY:
            raise KeyError(f"Unknown tool: {name}. Registered: {get_registered_names()}")
        out.append(_TOOL_REGISTRY[name]())
    return out
