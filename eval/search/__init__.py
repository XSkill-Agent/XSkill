"""
Search module for multimodal agent reasoning.

This module provides the SearchNode data structure for organizing
conversation history, images, and reasoning state during inference.
"""

from .tree import SearchNode

__all__ = [
    'SearchNode',
]
