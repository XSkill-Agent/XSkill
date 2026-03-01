"""
API caller utilities for vision API inference.
"""

import os
import time
import random
import json
import requests

# Error code classification: non-retryable (do not retry on these)
NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404}

# API configuration constants
API_TIMEOUT = 240  # API request timeout (seconds)
BASE_WAIT_TIME = 1.0  # Base wait time (seconds)
MAX_WAIT_TIME = 15.0  # Maximum wait time (seconds)
ROUND_ROBIN_429_WAIT_MIN = 5.0  # Round-Robin mode 429 error minimum wait time
ROUND_ROBIN_429_WAIT_MAX = 10.0  # Round-Robin mode 429 error maximum wait time
HIGH_TOKEN_THRESHOLD = 25000  # High token number warning threshold
DEFAULT_MAX_RETRIES = 2  # Default maximum retry count

# ====================== api config functions ======================================

def _build_payload(model_name: str, messages: list, sampling_params: dict, tools: list = None):
    """
    Build API request payload
    
    Returns:
        dict: Built payload
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling_params['temperature'],
        "top_p": sampling_params['top_p'],
        "max_tokens": sampling_params['max_tokens'],
    }
    
    # Add tools if provided (for function calling)
    if tools:
        payload["tools"] = tools
        if "o4-mini" not in model_name:
            payload["parallel_tool_calls"] = False
    
    return payload


def _add_reasoning_param(payload: dict, model_name: str, end_point: str = None):
    """
    Add reasoning parameters based on model type and API provider
    
    Args:
        payload: Dictionary to modify
        model_name: Model name
        api_name: API name identifier (for logging)
        end_point: API endpoint URL (for detecting provider, optional)
    """
    model_lower = model_name.lower()
    is_gemini_model = "gemini" in model_lower
    is_gpt_model = "gpt-5" in model_lower or "o1" in model_lower or "o3" in model_lower
    
    # Check if it is OpenRouter API
    is_openrouter = False
    if end_point:
        is_openrouter = "openrouter" in end_point.lower()
    
    if is_gemini_model:
        # OpenRouter Gemini models use max_tokens parameter (according to official documentation)
        if is_openrouter:
            reasoning_max_tokens = os.environ.get("REASONING_MAX_TOKENS")
            if reasoning_max_tokens and reasoning_max_tokens.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "max_tokens": int(reasoning_max_tokens),
                    "exclude": False
                }
        else:
            reasoning_effort = os.environ.get("REASONING_EFFORT")
            if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
                payload["reasoning"] = {
                    "effort": reasoning_effort,  # "xhigh", "high", "medium", "low", "minimal"
                    "exclude": False
                }
    elif is_gpt_model:
        # GPT models use OpenAI-style reasoning_effort (all providers support this)
        reasoning_effort = os.environ.get("REASONING_EFFORT")
        if reasoning_effort and reasoning_effort.lower() not in ["none", "false", ""]:
            payload["reasoning_effort"] = reasoning_effort  # "high", "medium", "low", "minimal"
    else:
        pass


def _make_api_request(end_point: str, headers: dict, payload: dict, api_name: str = "API"):
    """
    Send API request and return response
    
    Returns:
        tuple: (response, error_type) 
               response: requests.Response object or None
               error_type: "429", "timeout", "network", "other", None(Success)
    """
    try:
        response = requests.post(end_point, headers=headers, json=payload, timeout=API_TIMEOUT)
        if response.status_code == 429:
            return response, "429"
        return response, None
    except requests.exceptions.Timeout:
        print(f"[{api_name}] API timeout")
        return None, "timeout"
    except requests.exceptions.RequestException as e:
        print(f"[{api_name}] API call failed: {e}")
        return None, "network"
    except Exception as e:
        print(f"[{api_name}] Unexpected error: {e}")
        return None, "other"


def _parse_api_response(response, api_name: str, attempt: int = 0, max_attempts: int = 1):
    """
    Parse API response.

    Returns:
        tuple: (result, is_429, error_type)
               result: Parsed result (dict/str/None)
               is_429: Whether it is a 429 error (bool)
               error_type: Error type ("http_error", "empty_choices", "empty_content", "invalid_format", None)
    """
    # Handle HTTP errors
    if response.status_code != 200:
        error_info = None
        error_text = None
        try:
            error_info = response.json()
            error_text = str(error_info)
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API Error {response.status_code}{attempt_str}: {error_info}")
        except:
            error_text = response.text
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API Error {response.status_code}{attempt_str}: {response.text}")
        
        is_429 = (response.status_code == 429)
        return None, is_429, "http_error"
    
    # Parse successful response
    try:
        result = response.json()
    except json.JSONDecodeError as e:
        print(f"[{api_name}] Failed to parse JSON response: {e}")
        return None, False, "invalid_format"
    
    # Handle response content
    if "choices" in result and result["choices"]:
        # Security check: ensure choices[0] exists and is not None
        first_choice = result["choices"][0]
        if first_choice is None:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API returned None in choices[0]{attempt_str}")
            return None, False, "invalid_format"
        
        # Security check: ensure message field exists and is a dictionary
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API response missing 'message' field in choices[0]{attempt_str}")
            print(f"[{api_name}] choices[0] content: {first_choice}")
            return None, False, "invalid_format"
        
        message = first_choice["message"]
        
        # Security check: ensure message is not None and is a dictionary
        if message is None or not isinstance(message, dict):
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API returned invalid message type{attempt_str}: {type(message)}")
            return None, False, "invalid_format"
        
        # Check for function call (tool_calls field)
        if "tool_calls" in message and message["tool_calls"]:
            return message, False, None
        
        # Check for content (text response)
        content = message.get("content")
        if content and content.strip():
            # Return full message if there's reasoning_details to preserve
            if "reasoning_details" in message or "reasoning" in message:
                return message, False, None
            return content, False, None
        else:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] API returned empty content{attempt_str}")
            if max_attempts > 1:
                try:
                    print(f"[{api_name}] Empty message detail: {json.dumps(message, ensure_ascii=False)}")
                except Exception:
                    print(f"[{api_name}] Empty message detail: {message}")
            return None, False, "empty_content"
    else:
        # Handle empty choices array (model refused to generate)
        if "choices" in result and len(result["choices"]) == 0:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] Model refused to generate (empty choices){attempt_str}")
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            print(f"[{api_name}] Usage: prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}")
            
            # Check for safety/content filtering
            if "error" in result:
                print(f"[{api_name}] Error details: {result['error']}")
            
            # Check if prompt tokens are very high (possible token limit issue)
            if prompt_tokens > HIGH_TOKEN_THRESHOLD:
                print(f"[{api_name}] Warning: Very high prompt_tokens ({prompt_tokens}), may have hit token limit")
            
            print(f"[{api_name}] Possible causes: content filtering, safety policy, token limit, or model overload")
            return None, False, "empty_choices"
        else:
            attempt_str = f" on attempt {attempt + 1}/{max_attempts}" if max_attempts > 1 else ""
            print(f"[{api_name}] Invalid API response format{attempt_str}: {result}")
            return None, False, "invalid_format"


# ====================== call vision api ==============================================

def _try_single_attempt(api_key: str, end_point: str, model_name: str, messages: list, 
                        sampling_params: dict, api_name: str = "API", tools: list = None):
    """
    Try a single API attempt (no retries). Used for round-robin polling.
    
    Args:
        api_key: API key for authentication
        end_point: API endpoint URL
        model_name: Name of the model to use
        messages: List of message dictionaries
        sampling_params: Dictionary containing temperature, top_p, max_tokens
        api_name: Name identifier for logging
        tools: Optional list of tool declarations for function calling
        
    Returns:
        Tuple of (result, is_429, error_info): 
            result: dict or string or None
            is_429: bool indicating 429 error
            error_info: str with error details or None
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Build payload
    payload = _build_payload(model_name, messages, sampling_params, tools)
    _add_reasoning_param(payload, model_name, end_point)
    
    # Send request
    response, request_error = _make_api_request(end_point, headers, payload, api_name)
    
    # Handle 429 error: check first (429 may be returned through request_error="429" or response.status_code)
    if request_error == "429" or (response and response.status_code == 429):
        wait_time = ROUND_ROBIN_429_WAIT_MIN + random.uniform(0, ROUND_ROBIN_429_WAIT_MAX - ROUND_ROBIN_429_WAIT_MIN)
        print(f"[{api_name}] API rate limit hit (429), waiting {wait_time:.2f} seconds before trying other API")
        time.sleep(wait_time)
        return (None, True, f"{api_name}: HTTP 429 (rate limit)")
    
    # Handle other request errors (timeout, network error, etc.)
    if request_error:
        error_msg = f"{api_name}: {request_error}"
        print(f"[{api_name}] Request failed: {request_error}")
        return (None, False, error_msg)
    
    # Parse response
    result, is_429, error_type = _parse_api_response(response, api_name, attempt=0, max_attempts=1)
    
    # Build error information
    error_info = None
    if result is None:
        if error_type == "http_error":
            try:
                error_detail = response.json() if response else {}
                error_info = f"{api_name}: HTTP {response.status_code} - {error_detail.get('error', {}).get('message', str(error_detail))}"
            except:
                error_info = f"{api_name}: HTTP {response.status_code}"
        elif error_type:
            error_info = f"{api_name}: {error_type}"
    
    # Round-Robin mode: all errors are returned immediately, allowing the upper layer to switch API
    return (result, is_429, error_info)


def _try_single_api(api_key: str, end_point: str, model_name: str, messages: list, 
                    sampling_params: dict, max_retries: int, api_name: str = "API", tools: list = None):
    """
    Internal function to try a single API with full retry logic.
    
    Args:
        api_key: API key for authentication
        end_point: API endpoint URL
        model_name: Name of the model to use
        messages: List of message dictionaries
        sampling_params: Dictionary containing temperature, top_p, max_tokens
        max_retries: Maximum number of retry attempts
        api_name: Name identifier for logging (e.g., "Primary API" or "Fallback API")
        tools: Optional list of tool declarations for function calling
        
    Returns:
        Model response (string or dict with tool_calls), or None on failure
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    
    # Build payload (only build once)
    payload = _build_payload(model_name, messages, sampling_params, tools)
    _add_reasoning_param(payload, model_name, end_point)
    
    for attempt in range(max_retries):
        # Send request
        response, request_error = _make_api_request(end_point, headers, payload, api_name)
        
        # Handle request errors (timeout, network error, etc.)
        if request_error:
            if attempt < max_retries - 1:
                time.sleep(BASE_WAIT_TIME)
                continue
            return None
        
        # Handle 429 error: exponential backoff and retry
        if response.status_code == 429:
            wait_time = BASE_WAIT_TIME * (2 ** attempt) + random.uniform(0, 1)
            wait_time = min(wait_time, MAX_WAIT_TIME)
            print(f"[{api_name}] API rate limit hit on attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.2f} seconds before retrying.")
            time.sleep(wait_time)
            continue
        
        # Parse response
        result, is_429, error_type = _parse_api_response(response, api_name, attempt=attempt, max_attempts=max_retries)
        
        # Return successfully
        if result is not None:
            return result
        
        # Handle errors: determine whether to retry based on error type
        # empty_choices does not retry (model refused to generate, retry无效)
        if error_type == "empty_choices":
            return None
        
        # Non-retryable error codes (400, 401, 403, 404) do not retry
        if error_type == "http_error" and response and response.status_code in NON_RETRYABLE_STATUS_CODES:
            return None
        
        # Other errors: wait and retry
        if attempt < max_retries - 1:
            time.sleep(BASE_WAIT_TIME)
            continue
        else:
            return None
    
    return None


def call_vision_api(model_name: str, messages: list, sampling_params: dict, max_retries: int = None, tools: list = None):
    """
    Call the vision API with robust retry logic, including exponential backoff for rate limiting.
    Supports round-robin polling between primary and fallback APIs for faster response.
    
    Args:
        model_name: Name of the model to use
        messages: List of message dictionaries
        sampling_params: Dictionary containing temperature, top_p, max_tokens
        max_retries: Maximum number of retry attempts (shared between both APIs in round-robin)
        tools: Optional list of tool declarations for function calling (Gemini format)
        
    Returns:
        Model response (string or dict with tool_calls), or None on failure
    """
    # Use default retry count (if not specified)
    if max_retries is None:
        max_retries = DEFAULT_MAX_RETRIES
    
    # Get primary API configuration
    api_key_1 = os.environ.get("REASONING_API_KEY")
    end_point_1 = os.environ.get("REASONING_END_POINT")

    if not all([api_key_1, end_point_1]):
        raise ValueError("REASONING_API_KEY and REASONING_END_POINT must be set.")

    # Get fallback API configuration (optional)
    api_key_2 = os.environ.get("REASONING_API_KEY_2")
    end_point_2 = os.environ.get("REASONING_END_POINT_2")
    
    has_fallback = bool(api_key_2 and end_point_2)
    
    # Round-Robin polling: alternate between primary and fallback APIs
    if has_fallback:
        print(f"[API Round-Robin] Starting round-robin polling with {max_retries} total attempts (shared between both APIs)")
        consecutive_429_count = 0  # Track consecutive rounds where both APIs return 429 errors
        all_errors = []  # Collect all error information
        
        for attempt in range(max_retries):
            # Try primary API first in each Round-Robin
            result_1, is_429_1, error_1 = _try_single_attempt(api_key_1, end_point_1, model_name, messages, 
                                                     sampling_params, api_name="Primary API", tools=tools)
            if result_1 is not None:
                return result_1
            if error_1:
                all_errors.append(f"Round {attempt + 1}: {error_1}")
            
            # Try fallback API in each Round-Robin
            result_2, is_429_2, error_2 = _try_single_attempt(api_key_2, end_point_2, model_name, messages, 
                                                     sampling_params, api_name="Fallback API", tools=tools)
            if result_2 is not None:
                return result_2
            if error_2:
                all_errors.append(f"Round {attempt + 1}: {error_2}")
            
            # Both failed - check if both returned 429 errors
            if is_429_1 and is_429_2:
                consecutive_429_count += 1
                # If both APIs are rate-limited, wait longer before next round
                # Exponential backoff: 20s, 30s, 40s, max 60s
                extra_wait = min( consecutive_429_count * 5, 15)
                print(f"[API Round-Robin] Both APIs rate-limited (429), waiting {extra_wait} seconds before next round")
                if attempt < max_retries - 1:
                    time.sleep(extra_wait)
            else:
                # Reset counter if not both 429
                consecutive_429_count = 0
                # Normal wait between rounds
                if attempt < max_retries - 1:
                    time.sleep(1)
        
        # Build detailed error message
        error_summary = "All API attempts failed (both primary and fallback)"
        if all_errors:
            error_summary += f". Errors: {'; '.join(all_errors)}"
        print(f"[API Round-Robin] {error_summary}")
        return f"Error: {error_summary}"
    else:
        # No fallback API, use original retry logic with primary API only
        print(f"[API] No fallback API configured, using primary API with {max_retries} retries")
        result = _try_single_api(api_key_1, end_point_1, model_name, messages, sampling_params, 
                                max_retries, api_name="Primary API", tools=tools)
        if result is not None:
            return result
        else:
            print(f"[API] Primary API failed after {max_retries} attempts")
            return "Error: All API attempts failed (primary API only)"
