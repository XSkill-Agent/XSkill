#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Search tree data structures for inference.

This module provides the SearchNode class and related utilities for
representing and managing conversation state during multi-step reasoning.
"""

import copy
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from PIL import Image


@dataclass
class SearchNode:
    """
    Represents the current state in a multi-step reasoning process.
    
    Each node contains the complete state at a particular point in the
    reasoning process, including conversation history, images, and metadata.
    """
    
    # Core state information
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    image_map: Dict[str, Image.Image] = field(default_factory=dict)
    current_turn: int = 0
    
    # Tree structure
    parent: Optional['SearchNode'] = None
    children: List['SearchNode'] = field(default_factory=list)
    
    # Evaluation and scoring
    score: float = 0.0
    evaluation_analysis: str = ""
    is_final: bool = False
    final_answer: Optional[str] = None
    
    # Metadata
    node_id: str = ""
    depth: int = 0
    created_at: float = field(default_factory=time.time)
    
    # API-specific state
    api_conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    current_token_count: int = 0
    
    # Realtime experience learning (optional)
    pending_step_experiences: List[str] = field(default_factory=list)
    realtime_experiences_history: List[List[str]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize node after creation."""
        if not self.node_id:
            self.node_id = f"node_{int(time.time() * 1000000)}"
    
    def copy(self) -> 'SearchNode':
        """Create a deep copy of this node without parent/children relationships."""
        new_node = SearchNode(
            conversation_history=copy.deepcopy(self.conversation_history),
            image_map=copy.deepcopy(self.image_map),
            current_turn=self.current_turn,
            parent=None,  # Don't copy tree relationships
            children=[],  # Don't copy tree relationships
            score=self.score,
            evaluation_analysis=self.evaluation_analysis,
            is_final=self.is_final,
            final_answer=self.final_answer,
            node_id=f"{self.node_id}_copy_{int(time.time() * 1000000)}",
            depth=self.depth,
            created_at=time.time(),
            api_conversation_history=copy.deepcopy(self.api_conversation_history),
            current_token_count=self.current_token_count,
            pending_step_experiences=copy.deepcopy(self.pending_step_experiences),
            realtime_experiences_history=copy.deepcopy(self.realtime_experiences_history)
        )
        
        # Copy dynamic attributes (e.g., turn_offset, observations, image_size_used_list)
        for attr in ['turn_offset', 'observations', 'image_size_used_list', 'save_dir_override']:
            if hasattr(self, attr):
                try:
                    val = getattr(self, attr)
                    if isinstance(val, list):
                        setattr(new_node, attr, val.copy())
                    else:
                        setattr(new_node, attr, val)
                except Exception:
                    pass
        
        return new_node
    
    def add_child(self, child: 'SearchNode') -> None:
        """Add a child node and set its parent."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)
    
    def get_path_to_root(self) -> List['SearchNode']:
        """Get the path from this node to the root."""
        path = []
        current = self
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))
    
    def get_conversation_path(self) -> List[Dict[str, Any]]:
        """Get the complete conversation history from root to this node."""
        path = self.get_path_to_root()
        conversation = []
        
        for node in path:
            conversation.extend(node.conversation_history)
        
        return conversation
    
    def update_score(self, score: float, analysis: str = "") -> None:
        """Update the node's evaluation score and analysis."""
        self.score = score
        self.evaluation_analysis = analysis
    
    def mark_final(self, answer: str) -> None:
        """Mark this node as containing a final answer."""
        self.is_final = True
        self.final_answer = answer
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state for debugging."""
        return {
            'node_id': self.node_id,
            'depth': self.depth,
            'current_turn': self.current_turn,
            'score': self.score,
            'is_final': self.is_final,
            'final_answer': self.final_answer[:100] + "..." if self.final_answer and len(self.final_answer) > 100 else self.final_answer,
            'num_images': len(self.image_map),
            'conversation_length': len(self.conversation_history),
            'token_count': self.current_token_count,
            'has_parent': self.parent is not None,
            'num_children': len(self.children)
        }
    
    def to_trajectory_text(self) -> str:
        """Convert the node's conversation to readable trajectory text."""
        trajectory = ""
        for message in self.conversation_history:
            role = message.get('role', 'unknown')
            content = message.get('content', '')
            
            # Handle different content formats
            if isinstance(content, list):
                # Multi-modal content
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        text_parts.append(part.get('text', ''))
                    elif isinstance(part, dict) and part.get('type') == 'image_url':
                        text_parts.append('[Image]')
                content = ' '.join(text_parts)
            elif isinstance(content, str):
                # Simple text content
                pass
            else:
                content = str(content)
            
            trajectory += f"**{role}**: {content}\n\n"
        
        if self.final_answer:
            trajectory += f"**Final Answer**: {self.final_answer}\n"
        
        return trajectory
    
    def estimate_tokens(self) -> int:
        """Estimate the total token count for this node's conversation."""
        total_chars = 0
        for message in self.conversation_history:
            content = message.get('content', '')
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get('type') == 'text':
                        total_chars += len(part.get('text', ''))
        
        # Rough approximation: 1 token ≈ 4 characters for English text
        return total_chars // 4
    
    def can_expand(self, max_turns: int, max_images: int, max_tokens: int) -> bool:
        """Check if this node can be expanded further."""
        if self.is_final:
            return False
        
        if self.current_turn >= max_turns:
            return False
        
        if len(self.image_map) >= max_images:
            return False
        
        if self.estimate_tokens() >= max_tokens:
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of the node."""
        return f"SearchNode(id={self.node_id}, depth={self.depth}, score={self.score:.3f}, final={self.is_final})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"SearchNode(id={self.node_id}, depth={self.depth}, turn={self.current_turn}, "
                f"score={self.score:.3f}, final={self.is_final}, children={len(self.children)})")
