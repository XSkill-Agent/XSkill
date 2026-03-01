
import os
import time
import requests
from typing import Optional, Dict, List, Union, Any
from PIL import Image

from .experience_utils import image_to_base64


# --------- Constants ---------

# Retry configuration
MAX_RETRIES = 3

# Timeout configuration
API_TIMEOUT = 300  # Timeout for general API requests (seconds)
API_TIMEOUT_IMAGE = 300  # Timeout for multimodal API requests (seconds)

# Token configuration
MAX_TOKENS_DEFAULT = 12288  # Default max tokens for LLM calls


class ExperienceLLM:
    """Lightweight LLM adapter for experience generation via API."""

    def __init__(self, model_name: Optional[str] = None):
        # Unified naming: EXPERIENCE_MODEL_NAME (with fallback to old names for backward compatibility)
        self.model_name = model_name or os.environ.get("EXPERIENCE_MODEL_NAME") or os.environ.get("EXPERIENCE_MODEL") or os.environ.get("MODEL_NAME") or "gpt-4o"

    def chat(self, prompt: str, max_tokens: int = MAX_TOKENS_DEFAULT, temperature: Optional[float] = None, top_p: float = 1.0) -> str:
        t = temperature if temperature is not None else 0.6
        return self._call_with_fallback(
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            primary_api_name="Primary Experience API",
            fallback_api_name="Fallback Experience API"
        )

    def _try_single_experience_api(
        self, 
        api_key: str, 
        end_point: str, 
        user_content: Union[str, List[Dict[str, Any]]], 
        max_tokens: int, 
        temperature: float, 
        top_p: float, 
        max_retries: int, 
        api_name: str = "API",
        system_prompt: Optional[str] = None,
        timeout: int = API_TIMEOUT,
        return_placeholder_on_error: bool = False
    ) -> str:
        """
        Internal function to try a single Experience API with full retry logic.
        Supports both text-only and multimodal (text + image) inputs.
        
        Args:
            api_key: API key for authentication
            end_point: API endpoint URL
            user_content: Prompt text (str) or multimodal content list (List[Dict])
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_retries: Maximum number of retry attempts
            api_name: Name identifier for logging
            system_prompt: Optional system prompt (default: experience generation prompt)
            timeout: Request timeout in seconds
            return_placeholder_on_error: If True, return error placeholder instead of raising
            
        Returns:
            Response content string
            
        Raises:
            RuntimeError: If all retry attempts fail (unless return_placeholder_on_error=True)
        """
        # Default system prompt for experience generation
        if system_prompt is None:
            system_prompt = "You are an expert at summarizing visual reasoning trajectories and extracting generalizable experiences."

        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        
        base_wait_time = 1  # Base wait time in seconds for exponential backoff
        
        for attempt in range(max_retries):
            try:
                resp = requests.post(end_point, headers=headers, json=payload, timeout=timeout)
                
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if content:
                        return content
                    else:
                        error_msg = f"[{api_name}] API returned empty content on attempt {attempt + 1}/{max_retries}"
                        print(error_msg)
                        if attempt < max_retries - 1:
                            time.sleep(base_wait_time)
                            continue
                        else:
                            if return_placeholder_on_error:
                                return f"[Failed to generate: empty response]"
                            raise RuntimeError(f"[{api_name}] API returned empty content after {max_retries} attempts")
                
                # Handle errors
                try:
                    detail = resp.json()
                    error_msg = f"[{api_name}] Experience API error {resp.status_code} on attempt {attempt + 1}/{max_retries}: {detail}"
                except Exception:
                    detail = resp.text
                    error_msg = f"[{api_name}] Experience API error {resp.status_code} on attempt {attempt + 1}/{max_retries}: {detail}"
                
                print(error_msg)
                
                if attempt < max_retries - 1:
                    time.sleep(base_wait_time)
                    continue
                else:
                    if return_placeholder_on_error:
                        return f"[Failed to generate: {resp.status_code}]"
                    raise RuntimeError(error_msg)
                    
            except requests.exceptions.Timeout:
                error_msg = f"[{api_name}] API timeout on attempt {attempt + 1}/{max_retries}"
                print(error_msg)
                if attempt == max_retries - 1:
                    if return_placeholder_on_error:
                        return f"[Failed to generate: timeout]"
                    raise RuntimeError(f"[{api_name}] API timeout after {max_retries} attempts")
                time.sleep(base_wait_time)
            except requests.exceptions.RequestException as e:
                error_msg = f"[{api_name}] API call failed on attempt {attempt + 1}/{max_retries}: {e}"
                print(error_msg)
                if attempt == max_retries - 1:
                    if return_placeholder_on_error:
                        return f"[Failed to generate: {str(e)}]"
                    raise RuntimeError(f"[{api_name}] API call failed after {max_retries} attempts: {e}")
                time.sleep(base_wait_time)
            except RuntimeError:
                # Re-raise RuntimeError (from status code errors) unless placeholder mode
                if return_placeholder_on_error:
                    return f"[Failed to generate: API error]"
                raise
            except Exception as e:
                error_msg = f"[{api_name}] Unexpected error on attempt {attempt + 1}/{max_retries}: {e}"
                print(error_msg)
                if attempt == max_retries - 1:
                    if return_placeholder_on_error:
                        return f"[Error generating: {str(e)}]"
                    raise RuntimeError(f"[{api_name}] Unexpected error after {max_retries} attempts: {e}")
                time.sleep(base_wait_time)
        
        # Should not reach here, but handle just in case
        if return_placeholder_on_error:
            return f"[Failed to generate: all retry attempts exhausted]"
        raise RuntimeError(f"[{api_name}] All retry attempts exhausted")

    def _get_api_config(self, primary_key: str, primary_endpoint: str, 
                       fallback_key: str, fallback_endpoint: str) -> tuple:
        """
        Get API configuration from environment variables with fallback support.
        
        Args:
            primary_key: Environment variable name for primary API key
            primary_endpoint: Environment variable name for primary API endpoint
            fallback_key: Environment variable name for fallback API key
            fallback_endpoint: Environment variable name for fallback API endpoint
            
        Returns:
            Tuple of (api_key_1, end_point_1, api_key_2, end_point_2)
            Fallback values are None if not configured
        """
        api_key_1 = os.environ.get(primary_key) or os.environ.get("REASONING_API_KEY")
        end_point_1 = os.environ.get(primary_endpoint) or os.environ.get("REASONING_END_POINT")
        
        api_key_2 = os.environ.get(fallback_key) or os.environ.get("REASONING_API_KEY_2")
        end_point_2 = os.environ.get(fallback_endpoint) or os.environ.get("REASONING_END_POINT_2")
        
        return api_key_1, end_point_1, api_key_2, end_point_2

    def _normalize_endpoint(self, endpoint: str, require_chat_completions: bool = False) -> str:
        """
        Normalize API endpoint URL.
        
        Args:
            endpoint: API endpoint URL
            require_chat_completions: If True, ensure URL ends with /chat/completions
            
        Returns:
            Normalized endpoint URL
        """
        if require_chat_completions and not endpoint.endswith("/chat/completions"):
            if endpoint.endswith("/"):
                endpoint = endpoint + "chat/completions"
            else:
                endpoint = endpoint + "/chat/completions"
        return endpoint

    def _call_with_fallback(
        self,
        user_content: Union[str, List[Dict[str, Any]]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        system_prompt: Optional[str] = None,
        timeout: int = API_TIMEOUT,
        return_placeholder_on_error: bool = False,
        primary_api_name: str = "Primary API",
        fallback_api_name: str = "Fallback API",
        require_chat_completions: bool = False
    ) -> str:
        """
        Call API with primary and fallback support, reusing retry logic.
        
        Args:
            user_content: Text prompt (str) or multimodal content (List[Dict])
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            system_prompt: Optional system prompt
            timeout: Request timeout in seconds
            return_placeholder_on_error: If True, return placeholder instead of raising
            primary_api_name: Name for primary API logging
            fallback_api_name: Name for fallback API logging
            require_chat_completions: If True, normalize endpoint to /chat/completions
            
        Returns:
            Response content string
            
        Raises:
            RuntimeError: If both APIs fail (unless return_placeholder_on_error=True)
            ValueError: If primary API not configured
        """
        max_retries = MAX_RETRIES
        
        # Get API configuration
        api_key_1, end_point_1, api_key_2, end_point_2 = self._get_api_config(
            "EXPERIENCE_API_KEY", "EXPERIENCE_END_POINT",
            "EXPERIENCE_API_KEY_2", "EXPERIENCE_END_POINT_2"
        )
        
        if not api_key_1 or not end_point_1:
            if return_placeholder_on_error:
                return "[API not configured]"
            raise ValueError("EXPERIENCE_API_KEY/END_POINT or REASONING_API_KEY/END_POINT must be set")
        
        # Normalize primary endpoint
        end_point_1 = self._normalize_endpoint(end_point_1, require_chat_completions)
        
        # Try primary API
        try:
            return self._try_single_experience_api(
                api_key_1, end_point_1, user_content, max_tokens,
                temperature, top_p, max_retries,
                api_name=primary_api_name,
                system_prompt=system_prompt,
                timeout=timeout,
                return_placeholder_on_error=return_placeholder_on_error
            )
        except RuntimeError as e:
            # Primary API failed, try fallback if configured
            if api_key_2 and end_point_2:
                end_point_2 = self._normalize_endpoint(end_point_2, require_chat_completions)
                print(f"[API Fallback] {primary_api_name} failed, switching to {fallback_api_name}...")
                try:
                    return self._try_single_experience_api(
                        api_key_2, end_point_2, user_content, max_tokens,
                        temperature, top_p, max_retries,
                        api_name=fallback_api_name,
                        system_prompt=system_prompt,
                        timeout=timeout,
                        return_placeholder_on_error=return_placeholder_on_error
                    )
                except RuntimeError as e2:
                    if return_placeholder_on_error:
                        return f"[Failed: both APIs failed]"
                    raise RuntimeError(f"Both {primary_api_name} and {fallback_api_name} failed. Primary: {e}. Fallback: {e2}")
            else:
                if return_placeholder_on_error:
                    return f"[Failed: {str(e)}]"
                raise RuntimeError(f"{primary_api_name} failed and no fallback configured: {e}")

    def chat_with_image(
        self, 
        prompt: str, 
        image: Union[Image.Image, List[Image.Image]],
        max_tokens: int = MAX_TOKENS_DEFAULT, 
        temperature: Optional[float] = None,
        top_p: float = 1.0,
        system_prompt: Optional[str] = None,
        return_placeholder_on_error: bool = True
    ) -> str:
        """
        Chat with image input(s), reusing retry logic and error handling.
        
        Args:
            prompt: Text prompt describing what to do with the image(s)
            image: PIL Image or list of PIL Images to process
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (defaults to EXPERIENCE_TEMPERATURE or 0.6)
            top_p: Top-p sampling parameter
            system_prompt: Optional system prompt (default: image description prompt)
            return_placeholder_on_error: If True, return error placeholder instead of raising
            
        Returns:
            Response content string (or error placeholder if return_placeholder_on_error=True)
            
        Raises:
            RuntimeError: If both APIs fail (unless return_placeholder_on_error=True)
        """
        t = temperature if temperature is not None else 0.6
        # Default system prompt for image description
        if system_prompt is None:
            system_prompt = "You are an expert at analyzing images and providing detailed visual descriptions."
        
        # Normalize to list
        if isinstance(image, Image.Image):
            images = [image]
        else:
            images = image
        
        # Build multimodal content: text first, then all images
        user_content = [{"type": "text", "text": prompt}]
        
        for img in images:
            base64_image = image_to_base64(img)
            user_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            })
        
        # Use unified call_with_fallback method
        return self._call_with_fallback(
            user_content=user_content,
            max_tokens=max_tokens,
            temperature=t,
            top_p=top_p,
            system_prompt=system_prompt,
            timeout=API_TIMEOUT_IMAGE,  # Shorter timeout for image processing
            return_placeholder_on_error=return_placeholder_on_error,
            primary_api_name="Primary Multimodal API",
            fallback_api_name="Fallback Multimodal API",
            require_chat_completions=True  # Multimodal API requires /chat/completions endpoint
        )

