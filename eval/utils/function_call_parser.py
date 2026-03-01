"""
Function call parser for OpenAI-compatible API responses.
Parses tool calls and answer tags from model outputs.
"""

import re
import json
from typing import Tuple, Any, Dict, Union


def parse_function_call_response(response: Union[str, Dict], text_content: str = None) -> Tuple[str, Any]:
    """
    Parse API response supporting OpenAI function calling and text responses.
    
    Args:
        response: API response content, can be:
            - dict: Response object containing tool_calls
            - str: Plain text response
        text_content: Optional text content (used when response is dict)
        
    Returns:
        (action_type, data) tuple:
        - action_type: "function_call", "answer", "text", "error"
        - data: Parsed data
        
    Examples:
        >>> # OpenAI format function call
        >>> response = {"tool_calls": [{"function": {"name": "web_search", "arguments": '{"query": "test"}'}}]}
        >>> parse_function_call_response(response)
        ("function_call", {"tool_name": "web_search", "parameters": {"query": "test"}})
        
        >>> # Text response with <answer> tag
        >>> response = "Let me think... <answer>42</answer>"
        >>> parse_function_call_response(response)
        ("answer", "42")
    
    Reference:
        https://platform.openai.com/docs/guides/function-calling
    """
    
    # Case 1: Response is dict (may contain function calls)
    if isinstance(response, dict):
        # Check OpenAI format tool_calls
        if "tool_calls" in response and response["tool_calls"]:
            tool_calls = response["tool_calls"]
            
            # Only take the first tool call
            if isinstance(tool_calls, list) and len(tool_calls) > 0:
                if len(tool_calls) > 1:
                    print("[Function Call Parser] Warning: Multiple tool calls detected, using only the first one")
                
                first_call = tool_calls[0]
                function_data = first_call.get("function", {})
                
                tool_name = function_data.get("name", "")
                # OpenAI uses JSON string for arguments
                arguments = function_data.get("arguments", "{}")
                
                # Parse arguments (may be string or dict)
                if isinstance(arguments, str):
                    try:
                        parameters = json.loads(arguments)
                    except json.JSONDecodeError:
                        return "error", f"Invalid JSON in arguments: {arguments}"
                else:
                    parameters = arguments
                
                if not tool_name:
                    return "error", "Tool call missing 'name' field"
                
                return "function_call", {
                    "tool_name": tool_name,
                    "parameters": parameters
                }
        
        # If no function call, check for text content
        if text_content:
            response = text_content
        elif "content" in response:
            response = response["content"]
        elif "text" in response:
            response = response["text"]
        else:
            # No parseable content
            return "text", ""
    
    # Case 2: Response is string
    if isinstance(response, str):
        # Check for <answer> tag (for final answer)
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            return "answer", answer_match.group(1).strip()
        
        # Plain text response
        return "text", response
    
    # Unknown format
    return "error", f"Unknown response format: {type(response)}"
