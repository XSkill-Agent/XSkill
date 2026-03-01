"""
Context management utilities for inference.
Handles conversation history and token estimation.
"""

from PIL import Image
import math
import io
import base64
import re


def estimate_tokens(text: str) -> int:
    """
    Rough estimation of token count for context management.
    
    Args:
        text: Input text string
        
    Returns:
        Estimated token count
    """
    return len(text) // 4


def pil_to_base64_data_uri(pil_img, max_pixels: int = 2000000, min_pixels: int = 40000, quality: int = 85):
    """
    Converts a PIL image to a base64 data URI for API calls.
    Uses JPEG compression with quality=85 (aligned with Baseline framework).
    
    Args:
        pil_img: PIL Image object
        max_pixels: Maximum number of pixels (default: 2000000)
        min_pixels: Minimum number of pixels (default: 40000)
        quality: JPEG quality 1-100 (default: 85, balance between quality and size)
    
    Returns:
        Base64 data URI string
    """
    # Handle RGBA images before process_image (white background for transparency)
    if pil_img.mode == 'RGBA':
        background = Image.new('RGB', pil_img.size, (255, 255, 255))
        background.paste(pil_img, mask=pil_img.split()[3])  # Use alpha channel as mask
        pil_img = background
    
    # Apply resolution limiting (aligned with Baseline framework)
    processed_img = process_image(pil_img, max_pixels=max_pixels, min_pixels=min_pixels)
    
    # Save as JPEG with quality control
    buffered = io.BytesIO()
    processed_img.save(buffered, format="JPEG", quality=quality)
    base64_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/jpeg;base64,{base64_str}"


def process_image(image: Image.Image, max_pixels: int = 2000000, min_pixels: int = 40000, use_lanczos: bool = True) -> Image.Image:
    """
    Resizes an image to be within a specified pixel range.
    This logic is copied directly from `verl/workers/rollout/vllm_rollout/vllm_async_engine.py`
    to ensure behavioral consistency.

    Args:
        image: The input PIL image.
        max_pixels: The maximum number of pixels allowed.
        min_pixels: The minimum number of pixels allowed.
        use_lanczos: If True, use LANCZOS resampling (higher quality). Otherwise use NEAREST (faster).

    Returns:
        The resized PIL image.
    """
    resample_method = Image.Resampling.LANCZOS if use_lanczos else Image.Resampling.NEAREST
    
    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=resample_method)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=resample_method)

    if image.width < 28 or image.height < 28:
        resize_factor = 28 / min(image.width, image.height)
        width, height = int(image.width * resize_factor + 1), int(image.height * resize_factor + 1)
        image = image.resize((width, height), resample=resample_method)

    if image.width / image.height >= 200:
        width, height = image.width, int(image.width / 190 + 1)
        image = image.resize((width, height), resample=resample_method)

    if image.height / image.width >= 200:
        width, height = int(image.height / 190 + 1), image.height
        image = image.resize((width, height), resample=resample_method)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image
