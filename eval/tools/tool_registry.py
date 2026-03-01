"""
Tool registry, manage all available tools
Provide tool registration decorator and tool query functionality
"""

TOOL_REGISTRY = {}


def register_tool(name: str = None):
    """
    Tool registration decorator
    
    Args:
        name: Tool name, if not specified, use the class's name attribute
        
    Usage:
        @register_tool("my_tool")
        class MyTool(BaseTool):
            ...
    """
    def decorator(cls):
        tool_name = name or cls.name
        if tool_name in TOOL_REGISTRY:
            print(f"Warning: Tool '{tool_name}' already exists, overwriting")
        TOOL_REGISTRY[tool_name] = cls
        cls.name = tool_name
        return cls
    return decorator


def get_tool(name: str):
    """
    Get tool class
    
    Args:
        name: Tool name
        
    Returns:
        Tool class (uninstantiated)
        
    Raises:
        ValueError: If tool does not exist
    """
    if name not in TOOL_REGISTRY:
        available = list_tools()
        raise ValueError(
            f"Tool '{name}' not found in registry. "
            f"Available tools: {', '.join(available)}"
        )
    return TOOL_REGISTRY[name]


def list_tools():
    """
    List all registered tools
    
    Returns:
        Tool name list
    """
    return list(TOOL_REGISTRY.keys())


def get_tool_info(name: str):
    """
    Get tool's detailed information
    
    Args:
        name: Tool name
        
    Returns:
        Dictionary containing tool information
    """
    tool_cls = get_tool(name)
    return {
        "name": tool_cls.name,
        "description": tool_cls.description,
        "parameters": tool_cls.parameters
    }

