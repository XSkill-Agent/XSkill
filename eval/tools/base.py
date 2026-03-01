"""
Base Tool Class - Defines the interface for all tools.
Base class for all tools. All tools must inherit this class and implement the call method.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union


class BaseTool(ABC):
    """Base class for all tools. All tools must inherit this class and implement the call method."""
    
    name: str = ""           # Tool name
    description: str = ""    # Tool description
    parameters: dict = {}    # Parameters definition (JSON Schema format)
    
    def __init__(self, config: Dict = None):
        """
        Initialize the tool
        
        Args:
            config: Tool configuration dictionary
        """
        self.config = config or {}
        if not self.name:
            raise ValueError(f"Tool must have a name")
    
    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, dict]:
        """
        Tool call interface
        
        Args:
            params: Tool parameters
            **kwargs: Additional parameters (e.g. images, context, etc.)
            
        Returns:
            Tool execution result
        """
        raise NotImplementedError
    
    def validate_params(self, params: dict) -> bool:
        """
        Validate parameters to ensure they meet requirements
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Whether validation passed
        """
        # Simple validation: check required parameters
        required = self.parameters.get("required", [])
        return all(key in params for key in required)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

