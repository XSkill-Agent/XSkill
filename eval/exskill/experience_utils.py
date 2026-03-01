"""Utility functions for the experience module."""

import os
import json
import base64
import io
from typing import Dict
from PIL import Image


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 data URI.
    
    Args:
        image: PIL Image to convert
        
    Returns:
        Base64-encoded string
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


# --------- Experience Library I/O ---------

def load_existing(path: str) -> Dict[str, str]:
    """Load existing experiences from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping experience IDs to experience text
    """
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, dict) and "experiences" in data:
            return data["experiences"]
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_library(path: str, experiences: Dict[str, str]):
    """Save experiences to a JSON file.
    
    Args:
        path: Path to save the JSON file
        experiences: Dictionary mapping experience IDs to experience text
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"experiences": experiences}, f, ensure_ascii=False, indent=2)


def load_experiences(path: str) -> Dict[str, str]:
    """Load experiences from a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Dictionary mapping experience IDs to experience text
    """
    if path and os.path.exists(path):
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
            if isinstance(data, dict) and "experiences" in data:
                return data["experiences"]
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def format_for_prompt(experiences: Dict[str, str], max_items: int = 32) -> str:
    """Format experiences for injection into prompts.
    
    Args:
        experiences: Dictionary mapping experience IDs to experience text
        max_items: Maximum number of experiences to include
        
    Returns:
        Formatted string for prompt injection
    """
    from prompts.experience_prompts_test_time import INJECTION_HEADER
    
    if not experiences:
        return ""
    items = list(experiences.items())[:max_items]
    bullets = "\n".join([f"- [{k}] {v}" for k, v in items])
    return INJECTION_HEADER.format(bullets=bullets)

