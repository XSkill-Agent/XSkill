"""
Model caller for API-based inference (vision API + function calling).
"""

import os
import time
from search.tree import SearchNode
from utils.function_call_parser import parse_function_call_response
from utils.result_utils import save_trajectory
from engine.api_caller import call_vision_api
from engine.api_tool_handler import APIToolHandler

def _extract_reasoning_text(reasoning_details):
    """
    Extract human-readable reasoning text from reasoning_details structure.
    
    OpenRouter reasoning_details can be:
    - A list of reasoning detail objects with 'text', 'summary', or 'data' fields
    - A simple string
    - A dict with nested content
    
    Returns: Concatenated reasoning text or None
    """
    if not reasoning_details:
        return None
    
    # If it's already a string, return it
    if isinstance(reasoning_details, str):
        return reasoning_details
    
    # If it's a list (OpenRouter format)
    if isinstance(reasoning_details, list):
        texts = []
        for detail in reasoning_details:
            if isinstance(detail, dict):
                # Try different fields
                if "text" in detail:
                    texts.append(detail["text"])
                elif "summary" in detail:
                    texts.append(detail["summary"])
                elif "content" in detail:
                    texts.append(str(detail["content"]))
            elif isinstance(detail, str):
                texts.append(detail)
        return "\n".join(texts) if texts else None
    
    # If it's a dict
    if isinstance(reasoning_details, dict):
        if "text" in reasoning_details:
            return reasoning_details["text"]
        elif "content" in reasoning_details:
            return str(reasoning_details["content"])
        elif "summary" in reasoning_details:
            return reasoning_details["summary"]
    
    return None


def create_model_caller(args, sampling_params, save_dir=None):
    """
    Create a model caller function for API-based inference.
    
    Args:
        args: Command line arguments
        sampling_params: Sampling parameters for the model
        save_dir: Directory to save trajectory information
        
    Returns:
        Function that can be called with a SearchNode to get model response
    """
    # Create unified tool handler (shared across all calls)
    tool_handler = APIToolHandler(args, save_dir)
    
    # Build tools schema once (for function calling)
    # Use OpenAI format for compatibility with most APIs
    tools_schema = None
    
    # Check if function calling is enabled via environment variable (default: enabled)
    enable_fc = os.environ.get("ENABLE_FUNCTION_CALLING", "true").lower()
    function_calling_enabled = enable_fc not in ("false", "0", "no", "off")
        
    if function_calling_enabled:
        try:
            from utils.tool_schema_builder import build_openai_tools_schema
            from tools import list_tools
            
            # Get all available tools from the registry (automatically discovers all registered tools)
            all_tools = list_tools()
            
            # Filter tools based on ENABLED_TOOLS environment variable
            enabled_tools_env = os.environ.get("ENABLED_TOOLS", "").strip()
            if enabled_tools_env:
                # Parse comma-separated list and strip whitespace
                enabled_tools_list = [tool.strip() for tool in enabled_tools_env.split(",") if tool.strip()]
                # Filter to only include tools that are both in all_tools and in enabled_tools_list
                tool_names = [tool for tool in enabled_tools_list if tool in all_tools]
                # Warn about tools in ENABLED_TOOLS that are not registered
                missing_tools = [tool for tool in enabled_tools_list if tool not in all_tools]
                if missing_tools:
                    print(f"[Function Calling] Warning: Tools in ENABLED_TOOLS not found in registry: {missing_tools}")
            else:
                # If ENABLED_TOOLS is not set, include all tools (backward compatibility)
                tool_names = all_tools
                print(f"[Function Calling] ENABLED_TOOLS not set, using all registered tools: {tool_names}")
            
            tools_schema = build_openai_tools_schema(tool_names)

        except Exception as e:
            print(f"[Function Calling] Warning: Failed to build tools schema: {e}")
            tools_schema = None
    else:
        print(f"[Function Calling] Disabled (ENABLE_FUNCTION_CALLING={enable_fc})")
    
    def model_caller(node: SearchNode):
        """
        Call the vision API and process the response (text or function call).
        
        Args:
            node: SearchNode containing conversation history and images
            
        Returns:
            str: Response text; None on failure; or "Error: ..." string when API returns error.
        """
        try:
            # Use the node's API conversation history
            messages = node.api_conversation_history.copy()

            # Call the model once per step to obtain assistant text
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                # Pass tools for function calling
                response = call_vision_api(args.model_name, messages, sampling_params, tools=tools_schema)

                # Check if response is an error string
                if isinstance(response, str) and response.startswith("Error:"):
                    print(f"[Model Caller] API returned error: {response}")
                    # Return error string to be handled by caller
                    return response

                # Response can be string (text) or dict (function call)
                if response is not None:
                    if isinstance(response, str) and response.strip():
                        break
                    elif isinstance(response, dict):
                        break
                
                if attempt < max_retries - 1:
                    print(f"Warning: Empty response on attempt {attempt + 1}, retrying...")
                    time.sleep(1)
                else:
                    print(f"Error: Failed to get valid response after {max_retries} attempts")

            if response is None:
                return None
            
            # Extract text content and reasoning for logging and conversation history
            if isinstance(response, dict):
                response_text = response.get("content") or ""
            else:
                response_text = response if isinstance(response, str) else ""
            
            reasoning_details = None
            reasoning_text = None
            
            if isinstance(response, dict):
                # Check for reasoning_content first (direct reasoning text field)
                # reasoning_content is in the message object (choices[0].message.reasoning_content)
                if "reasoning_content" in response:
                    reasoning_text = response.get("reasoning_content")
                    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
                        reasoning_text = None

                # OpenRouter uses 'reasoning_details' not 'reasoning'
                if not reasoning_text:
                    reasoning_details = response.get("reasoning_details")
                    # Also check 'reasoning' for backward compatibility
                    if not reasoning_details:
                        reasoning_details = response.get("reasoning")
                    
                    # Extract readable reasoning text from reasoning_details
                    if reasoning_details:
                        reasoning_text = _extract_reasoning_text(reasoning_details)
                
                # Fallback: check if there's a simple 'reasoning' string field
                if not reasoning_text and isinstance(response.get("reasoning"), str):
                    reasoning_text = response.get("reasoning")
                
                if "tool_calls" in response and not reasoning_text and response_text and response_text.strip():
                    reasoning_text = response_text.strip()
                    print(f"[Reasoning] Using content field as reasoning_text: {len(reasoning_text)} chars")
            
            is_tool_call_response = isinstance(response, dict) and "tool_calls" in response
            tool_calls_list = response.get("tool_calls", []) if is_tool_call_response else []
            
            # Update the node with the response
            if is_tool_call_response:
                first_tool_call_only = [tool_calls_list[0]] if tool_calls_list else []
                assistant_msg = {
                    "role": "assistant", 
                    "content": response_text, 
                    "tool_calls": first_tool_call_only
                }
                # Preserve reasoning_details for OpenRouter Gemini models (CRITICAL for thought_signature)
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details
                    print(f"[Reasoning] Preserved reasoning_details in tool call message")
                node.api_conversation_history.append(assistant_msg)
                node.conversation_history.append({"role": "assistant", "content": f"[Function Call] {response_text}"})
            else:
                # Text response
                assistant_msg = {"role": "assistant", "content": response_text}
                # Preserve reasoning_details for OpenRouter Gemini models
                if reasoning_details:
                    assistant_msg["reasoning_details"] = reasoning_details
                    print(f"[Reasoning] Preserved reasoning_details in text message")
                node.api_conversation_history.append(assistant_msg)
                node.conversation_history.append({"role": "assistant", "content": response_text})
            
            # Get turn_offset once for all trajectory logging
            _turn_offset = getattr(node, 'turn_offset', 0)
            global_turn_idx = _turn_offset + node.current_turn
            
            # Save trajectory step if save_dir is provided
            if save_dir:
                if isinstance(response, dict) and "tool_calls" in response:
                    tool_calls = response.get("tool_calls", [])
                    tool_names = []
                    if tool_calls:
                        for tc in tool_calls[:1]:  # Only first tool call
                            func_info = tc.get("function", {})
                            tool_names.append(func_info.get("name", "unknown"))
                    
                    # Build display text: reasoning + tool call indicator
                    display_parts = []
                    if reasoning_text and reasoning_text.strip():
                        display_parts.append(reasoning_text.strip())
                    elif response_text and response_text.strip():
                        # Use content field if reasoning_text is not available
                        display_parts.append(response_text.strip())
                    
                    if tool_names:
                        display_parts.append(f"\n[Calling tools: {', '.join(tool_names)}]")
                    
                    # If no reasoning or content, at least show tool call indicator
                    display_text = "\n\n".join(display_parts) if display_parts else (f"\n[Calling tools: {', '.join(tool_names)}]" if tool_names else "")
                else:
                    # Text response: use reasoning_text if available, otherwise response_text
                    display_text = reasoning_text if reasoning_text else response_text
                
                assistant_turn_for_traj = {
                    "turn_idx": global_turn_idx,
                    "text_output": display_text,
                    "node_id": getattr(node, 'node_id', ''),
                }
                save_trajectory(save_dir,assistant_turn_for_traj)
            
            # Parse response using new function call parser
            action, data = parse_function_call_response(response, text_content=response_text)
            
            if action == "function_call" or action == "tool_call":
                # Handle function calls using unified tool handler
                tool_name = data.get("tool_name", "")
                parameters = data.get("parameters", {})
                tool_call_id = data.get("tool_call_id")
                
                if not tool_call_id and node.api_conversation_history:
                    last_msg = node.api_conversation_history[-1]
                    if isinstance(last_msg, dict) and last_msg.get("role") == "assistant" and "tool_calls" in last_msg:
                        tool_calls = last_msg["tool_calls"]
                        if tool_calls:
                            tool_call_id = tool_calls[0].get("id", "call_0")
                    
                # Execute tool call using unified handler
                result = tool_handler.execute_tool_call(
                    tool_name, parameters, node, global_turn_idx, tool_call_id
                )
                
                # Update node.api_conversation_history with feedback messages
                if result.get('feedback_messages'):
                    node.api_conversation_history.extend(result['feedback_messages'])
                
                if result.get('skip_processing'):
                    node.current_turn += 1
                    return response_text
            
            elif action == "answer":
                node.final_answer = data
            
            # Update turn counter
            node.current_turn += 1
            
            # Return text content for compatibility
            return response_text
            
        except Exception as e:
            import traceback
            print(f"Error in model caller: {e}")
            traceback.print_exc()
            return None
    
    return model_caller
