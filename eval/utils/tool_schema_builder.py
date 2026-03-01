"""
Tool schema builder for OpenAI function calling format.

Converts tools from the registry to OpenAI-compatible function calling schemas.
"""

from typing import List, Dict, Any, Optional

try:
    from tools import get_tool_info, TOOL_REGISTRY
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    print("Warning: Tools module not available in tool_schema_builder")


def build_openai_tools_schema(tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Generate OpenAI Function Calling tools parameter from tool registry.
    
    Compatible with OpenAI, Anthropic, and other OpenAI-compatible APIs.
    
    Args:
        tool_names: List of tool names to include. If None, includes all registered tools
        
    Returns:
        OpenAI API tools format: [{"type": "function", "function": {...}}, ...]
        
    Example:
        >>> tools = build_openai_tools_schema(["web_search"])
        >>> # Returns:
        >>> # [
        >>> #     {
        >>> #         "type": "function",
        >>> #         "function": {
        >>> #             "name": "web_search",
        >>> #             "description": "Search the web...",
        >>> #             "parameters": {...}
        >>> #         }
        >>> #     }
        >>> # ]
    
    Reference:
        https://platform.openai.com/docs/guides/function-calling
    """
    if not TOOLS_AVAILABLE:
        print("Warning: Tools not available, returning empty schema")
        return []
    
    # Use all registered tools if none specified
    if tool_names is None:
        tool_names = list(TOOL_REGISTRY.keys())
    
    tools = []
    
    for tool_name in tool_names:
        try:
            tool_info = get_tool_info(tool_name)
            if not tool_info:
                print(f"Warning: Tool '{tool_name}' not found in registry, skipping")
                continue
            
            # Build OpenAI format tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_info["name"],
                    "description": tool_info["description"],
                }
            }
            
            # Add parameters if present
            if tool_info.get("parameters"):
                tool_def["function"]["parameters"] = tool_info["parameters"]
            
            tools.append(tool_def)
            
        except Exception as e:
            print(f"Warning: Failed to build schema for tool '{tool_name}': {e}")
            continue
    
    return tools
