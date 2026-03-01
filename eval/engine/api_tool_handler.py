"""
Unified tool handler for API-based inference.
Handles all tool calls and directly updates SearchNode state.
"""

import os
import re
import json
import shutil
from PIL import Image
from utils.context_utils import process_image
from utils.context_utils import pil_to_base64_data_uri, estimate_tokens
from utils.result_utils import save_trajectory
from qwen_vl_utils import fetch_image # type: ignore

# Import tool support
try:
    from tools import get_tool
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False
    print("Warning: Tools module not available in API tool handler")

class APIToolHandler:
    """
    Unified tool handler for API-based inference.
    All methods directly modify the SearchNode object.
    """
    
    def __init__(self, args, save_dir, tool_instances=None):
        """
        Initialize the tool handler.
        
        Args:
            args: Command line arguments
            save_dir: Directory to save trajectory information
            tool_instances: Optional dict of cached tool instances
        """
        self.args = args
        self.save_dir = save_dir
        self.tool_instances = tool_instances or {}
        self.tool_configs = getattr(args, 'tool_configs', {})
        
        # Tool call counting and limits
        self.tool_call_counts = {}  # Track number of calls per tool
        image_search_max_calls = getattr(args, 'image_search_max_calls', None)
        web_search_max_calls = getattr(args, 'web_search_max_calls', None)
        
        # Set default limits if not specified
        if image_search_max_calls is None:
            image_search_max_calls = 3  # Default: 3 calls
        if web_search_max_calls is None:
            web_search_max_calls = 5  # Default: 5 calls
        
        self.tool_call_limits = {
            'image_search': image_search_max_calls,
            'web_search': web_search_max_calls,
        }
    
    def _determine_image_name(self, tool_name, image_ref, alt_text, node):
        """
        Unify image naming for all tools.
        Tools should generate the final name internally, and the processing layer only handles parsing and preservation.
        
        Args:
            tool_name: Tool name
            image_ref: Image reference path
            alt_text: Alt text in Markdown
            node: SearchNode object
            
        Returns:
            str: Determined image name
        """
        # Remove possible extension
        name_without_ext = os.path.splitext(alt_text)[0]
        
        # Check if it's already in the final format (generated internally by tools)
        # Supported formats: tool_image_N, observation_N
        if re.match(r'^(tool_image_\d+|observation_\d+)$', name_without_ext):
            return name_without_ext
        
        # Fallback logic (should not reach here, but kept as safety net)
        print(f"[Warning] Tool {tool_name} did not generate final image name '{alt_text}', using fallback")
        tool_image_count = len([k for k in node.image_map.keys() if k.startswith('tool_image_')])
        tool_image_count += 1
        return f'tool_image_{tool_image_count}'
    
    def process_tool_output_for_images(self, tool_result, tool_name, node):
        """
        Process images from tool output and update node.image_map.
        
        Args:
            tool_result: Tool return text result
            tool_name: Name of the tool that generated the result
            node: SearchNode object (directly modified)
            
        Returns:
            tuple: (new_images_list, processed_result_text)
                - new_images_list: [(image_name, PIL.Image), ...] newly loaded images
                - processed_result_text: Processed result text (paths replaced with placeholders)
        """

        if tool_name in ['image_search', 'visit']:
            if tool_name == 'image_search':
                print(f"[Image Processing] Skipping image extraction for {tool_name} (returns URLs only, no markdown images)")
            else:  # visit
                print(f"[Image Processing] Skipping image extraction for {tool_name} (returns text summary, images should not be extracted)")
            return [], tool_result
        
        new_images = []
        processed_text = tool_result
        
        # Find Markdown format images: ![alt](path) or ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        matches = re.findall(image_pattern, tool_result)
        
        for alt_text, image_ref in matches:
            try:
                # Determine if it's a local path or URL
                if image_ref.startswith('http://') or image_ref.startswith('https://'):
                    # URL - use fetch_image
                    pil_image = fetch_image({'image': image_ref, 'max_pixels': self.args.max_pixels})
                else:
                    # Local path
                    if not os.path.isabs(image_ref):
                        # Relative path, try to resolve relative to save_dir
                        image_ref = os.path.join(self.save_dir, image_ref)
                    
                    if os.path.exists(image_ref):
                        pil_image = Image.open(image_ref)
                    else:
                        print(f"Warning: Image file not found: {image_ref}")
                        continue
                
                # Process image size
                original_size = pil_image.width * pil_image.height
                processed_image = process_image(pil_image, self.args.max_pixels, self.args.min_pixels)
                processed_size = processed_image.width * processed_image.height
                
                # Log image processing info for debugging
                if original_size != processed_size:
                    print(f"[Image Processing] Resized: {pil_image.width}x{pil_image.height} ({original_size:,}px) -> {processed_image.width}x{processed_image.height} ({processed_size:,}px)")
                else:
                    print(f"[Image Processing] No resize needed: {processed_image.width}x{processed_image.height} ({processed_size:,}px)")
                
                # Determine image name based on tool type
                image_name = self._determine_image_name(tool_name, image_ref, alt_text, node)
                
                # Add to result list and mapping (store original PIL object for possible future cropping)
                new_images.append((image_name, processed_image))
                node.image_map[image_name] = pil_image  # Store original PIL object
                
                # Extract original filename (if there's path information)
                original_filename = None
                if os.path.isfile(image_ref):
                    original_filename = os.path.basename(image_ref)
                elif '/' in image_ref or '\\' in image_ref:
                    # Extract filename from path
                    original_filename = os.path.basename(image_ref)
                
                # Create alias file for code_interpreter images (default behavior)
                if original_filename:
                    # Save alias link to standard tool_image_k.jpg
                    original_path = os.path.join(self.save_dir, original_filename)
                    standard_path = os.path.join(self.save_dir, f"{image_name}.jpg")
                    
                    if not os.path.exists(original_path) and os.path.exists(standard_path):
                        try:
                            shutil.copy(standard_path, original_path)
                            print(f"[Image Alias] Created alias: {original_filename} -> {image_name}.jpg")
                        except Exception as e:
                            print(f"[Image Alias] Failed to create alias: {e}")
                
                # Replace reference in text, preserve original filename info
                original_markdown = f'![{alt_text}]({image_ref})'
                if original_filename:
                    # Include filename info: model can access via variable or filename
                    replacement = f'[Image: {image_name}, file: {original_filename}]'
                else:
                    replacement = f'[Image: {image_name}]'
                processed_text = processed_text.replace(original_markdown, replacement)
                
                print(f"[Image Processing] Loaded {image_name} from tool output")
                
            except Exception as e:
                print(f"Error loading image from {image_ref}: {e}")
                continue
        
        return new_images, processed_text
    
    def handle_image_search_reference(self, parameters, node):
        """
        Handle image_search tool's image reference resolution.
        Directly modifies parameters['image_url'].
        
        Supports both URL format (from WebWatcher-aligned image_search) and local file references.
        
        Args:
            parameters: Tool parameters dict (directly modified)
            node: SearchNode object
        """
        if 'image_url' not in parameters:
            return
        
        image_ref = parameters['image_url']
        
        # Case 0: If it's already a URL (http:// or https://), use it directly
        # This supports the new WebWatcher-aligned format where image_search returns URLs
        if image_ref.startswith('http://') or image_ref.startswith('https://'):
            print(f"[Image Reference] Using URL directly: {image_ref[:80]}...")
            parameters['image_url'] = image_ref
            return
        
        # Remove extension (if any) to get key name
        image_key = os.path.splitext(image_ref)[0]  # "tool_image_1.jpg" -> "tool_image_1"
        
        found_image = False
        
        # Case 1: In node.image_map (object in memory, use first)
        if image_key in node.image_map:
            img = node.image_map[image_key]
            file_path = os.path.join(self.save_dir, f"{image_key}.jpg")
            # Ensure file exists (save if not exists)
            if not os.path.exists(file_path):
                img.save(file_path, 'JPEG', quality=95)
                print(f"[Image Saved] {image_key} -> {file_path}")
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using {image_key} from node.image_map: {file_path}")
            found_image = True
        
        # Case 2a: Not in node.image_map, try to find tool_image_k.jpg in save_dir
        if not found_image and os.path.exists(os.path.join(self.save_dir, f"{image_key}.jpg")):
            file_path = os.path.join(self.save_dir, f"{image_key}.jpg")
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using existing file: {file_path}")
            found_image = True
        
        # Case 2b: Try to find original filename alias (e.g., statue_crop.png)
        if not found_image and os.path.exists(os.path.join(self.save_dir, image_ref)):
            file_path = os.path.join(self.save_dir, image_ref)
            parameters['image_url'] = file_path
            print(f"[Image Reference] Using original filename alias: {file_path}")
            found_image = True
        
        # Case 3: Image not found, raise error
        if not found_image:
            error_msg = f"Image reference '{image_ref}' not found in node.image_map or save_dir. Available images: 'tool_image_k', 'observation_k' or original filenames. For URLs from image_search results, use the URL directly."
            print(f"[Error] {error_msg}")
            raise ValueError(error_msg)
    
    def get_or_create_tool(self, tool_name):
        """
        Get or create a tool instance (with caching).
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance
        """
        if tool_name not in self.tool_instances:
            tool_cls = get_tool(tool_name)
            
            # Load tool config if available
            tool_config = {}
            if tool_name in self.tool_configs:
                tool_config = self.tool_configs[tool_name]
            
            # Add args and save_dir to config for tools that need them.
            tool_config['args'] = self.args
            tool_config['save_dir'] = self.save_dir
            
            self.tool_instances[tool_name] = tool_cls(config=tool_config)
            print(f"[Tool Cache] Created new instance of {tool_name}")
        else:
            print(f"[Tool Cache] Reusing existing instance of {tool_name}")
        
        return self.tool_instances[tool_name]
    
    def prepare_tool_kwargs(self, tool_name, node):
        """
        Prepare tool kwargs based on tool name.
        
        Args:
            tool_name: Name of the tool
            node: SearchNode object
            
        Returns:
            dict: Tool kwargs
        """
        tool_kwargs = {}
        if tool_name in ('code_interpreter', 'zoom'):
            tool_kwargs['image_map'] = node.image_map.copy()
        if tool_name == 'image_search':
            tool_kwargs['save_dir'] = self.save_dir
        
        return tool_kwargs
    
    def execute_tool_call(self, tool_name, parameters, node, turn_idx, tool_call_id=None):
        """
        Execute a tool call and update node state.
        
        Args:
            tool_name: Name of the tool
            parameters: Tool parameters dict
            node: SearchNode object (directly modified)
            turn_idx: Current turn index for trajectory logging
            tool_call_id: Optional tool call ID for function calling format
            
        Returns:
            dict: {
                'tool_result': str,
                'processed_result': str,
                'new_images': [(image_name, PIL.Image), ...],
                'feedback_messages': [message_dict, ...]  # For updating api_conversation_history
            }
        """
        if not TOOLS_AVAILABLE:
            error_msg = "Tools are not available. Please install tool dependencies."
            print(f"[Tool Error] {error_msg}")
            feedback_msg = f"{error_msg}\n\nPlease provide your answer based on available information."
            return {
                'tool_result': '',
                'processed_result': '',
                'new_images': [],
                'feedback_messages': [{
                    "role": "tool",
                    "content": json.dumps({"error": feedback_msg}),
                    "tool_call_id": tool_call_id or "call_0"
                }],
                'error': True
            }
        
        print(f"[Tool Call] {tool_name} with params: {parameters}")
        
        # Check if tool call limit is exceeded
        tool_limit = self.tool_call_limits.get(tool_name, None)
        if tool_limit is not None:
            current_count = self.tool_call_counts.get(tool_name, 0)
            if current_count >= tool_limit:
                error_msg = f"Maximum number of '{tool_name}' calls ({tool_limit}) exceeded. Current count: {current_count}"
                print(f"[Tool Limit] {error_msg}")
                feedback_msg = f"{error_msg}\n\nPlease continue your reasoning process with different approach."
                feedback_messages = [{
                    "role": "tool",
                    "content": json.dumps({"error": feedback_msg}),
                    "tool_call_id": tool_call_id or "call_0"
                }]
                # Update node
                node.conversation_history.append({"role": "user", "content": feedback_msg})
                node.current_token_count += estimate_tokens(feedback_msg)
                return {
                    'tool_result': '',
                    'processed_result': '',
                    'new_images': [],
                    'feedback_messages': feedback_messages,
                    'error': True
                }
        
        try:
            # Get or create tool instance
            tool = self.get_or_create_tool(tool_name)
            
            # Handle image_search image references
            if tool_name == 'image_search' and 'image_url' in parameters:
                self.handle_image_search_reference(parameters, node)
            
            # Prepare tool kwargs
            tool_kwargs = self.prepare_tool_kwargs(tool_name, node)
            
            tool_result = tool.call(parameters, **tool_kwargs)
            
            # Save tool call to trajectory
            if self.save_dir:
                try:
                    tool_call_event = {
                        "turn_idx": turn_idx,
                        "tool_call": {
                            "tool_name": tool_name,
                            "parameters": parameters,
                            "result": tool_result
                        }
                    }
                    if hasattr(node, 'node_id') and node.node_id:
                        tool_call_event["node_id"] = node.node_id
                    save_trajectory(self.save_dir,tool_call_event)
                except Exception:
                    pass
            
            # Process images from tool output
            new_images, processed_result = self.process_tool_output_for_images(
                tool_result, tool_name, node
            )
            
            # Save tool-returned images to disk
            if new_images:
                for image_name, pil_image in new_images:
                    image_path = os.path.join(self.save_dir, f"{image_name}.jpg")
                    
                    if not os.path.exists(image_path):
                        pil_image.save(image_path)
                        print(f"[Image Saved] {image_name} -> {image_path}")
                    if self.save_dir:
                        try:
                            image_info_event = {
                                "turn_idx": turn_idx,
                                "tool_image": {
                                    "image_name": image_name,
                                    "source_tool": tool_name,
                                    "file_path": f"{image_name}.jpg"
                                }
                            }
                            if hasattr(node, 'node_id') and node.node_id:
                                image_info_event["node_id"] = node.node_id
                            save_trajectory(self.save_dir,image_info_event)
                        except Exception:
                            pass
            
            # Build feedback messages using native function calling format
            feedback_messages = []
            
            # Format result as JSON string
            result_content = json.dumps({"result": processed_result if new_images else tool_result})
            
            function_response = {
                "role": "tool",
                "content": result_content,
                "tool_call_id": tool_call_id or "call_0"
            }
            feedback_messages.append(function_response)
            
            # If there are new images, add them in a separate user message
            # Update tool call count first (only count successful calls)
            self.tool_call_counts[tool_name] = self.tool_call_counts.get(tool_name, 0) + 1
            
            # Add remaining calls info (only if current tool is restricted)
            usage_text = ""
            if tool_name in self.tool_call_limits:
                remaining_calls_info = []
                for tname, limit in self.tool_call_limits.items():
                    if limit is not None:
                        current = self.tool_call_counts.get(tname, 0)
                        remaining_calls_info.append(f"{tname} {current}/{limit}")
                
                if remaining_calls_info:
                    usage_text = f"\n\n[Tool Usage: {', '.join(remaining_calls_info)}]"
            
            if new_images:
                message_content = []
                total_base64_size = 0
                for image_name, pil_image in new_images:
                    # Convert to base64 and track size
                    base64_data = pil_to_base64_data_uri(pil_image)
                    base64_size = len(base64_data)
                    total_base64_size += base64_size
                    print(f"[Image Encoding] {image_name}: {pil_image.width}x{pil_image.height} -> base64 size: {base64_size:,} bytes ({base64_size/1024:.1f} KB)")
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": base64_data}
                    })
                print(f"[Image Encoding] Total base64 size for {len(new_images)} images: {total_base64_size:,} bytes ({total_base64_size/1024:.1f} KB, {total_base64_size/1024/1024:.2f} MB)")
                message_content.append({"type": "text", "text": f"Images from tool output: {processed_result}{usage_text}"})
                feedback_messages.append({"role": "user", "content": message_content})
                feedback_msg = f"Tool '{tool_name}' returned with images: {processed_result}{usage_text}"
            else:
                feedback_msg = f"Tool '{tool_name}' returned: {tool_result}{usage_text}"
            
            # Update node conversation history
            node.conversation_history.append({"role": "user", "content": feedback_msg})
            node.current_token_count += estimate_tokens(feedback_msg)
            
            return {
                'tool_result': tool_result,
                'processed_result': processed_result,
                'new_images': new_images,
                'feedback_messages': feedback_messages,
                'error': False,
                'skip_processing': False
            }
            
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            print(f"[Tool Error] {error_msg}")
            
            # Save error to trajectory
            if self.save_dir:
                try:
                    error_event = {
                        "turn_idx": turn_idx,
                        "tool_error": {
                            "tool_name": tool_name,
                            "error": str(e)
                        }
                    }
                    if hasattr(node, 'node_id') and node.node_id:
                        error_event["node_id"] = node.node_id
                    save_trajectory(self.save_dir,error_event)
                except Exception:
                    pass
            
            # Provide error feedback
            feedback_msg = f"{error_msg}\n\nPlease try a different approach or provide your answer based on available information."
            feedback_messages = [{
                "role": "tool",
                "content": json.dumps({"error": feedback_msg}),
                "tool_call_id": tool_call_id or "call_0"
            }]
            
            # Update node
            node.api_conversation_history.extend(feedback_messages)
            node.conversation_history.append({"role": "user", "content": feedback_msg})
            node.current_token_count += estimate_tokens(feedback_msg)
            
            return {
                'tool_result': '',
                'processed_result': '',
                'new_images': [],
                'feedback_messages': feedback_messages,
                'error': True
            }

