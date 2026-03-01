"""
工具模块初始化
自动导入所有工具并注册到工具注册表
"""

from .base import BaseTool
from .tool_registry import TOOL_REGISTRY, register_tool, get_tool, list_tools, get_tool_info

try:
    from .code_interpreter import CodeInterpreter
except ImportError:
    CodeInterpreter = None
    print("Warning: CodeInterpreter not available")

try:
    from .web_search import WebSearch
except ImportError:
    WebSearch = None
    print("Warning: WebSearch not available")

try:
    from .visit import Visit
except ImportError:
    Visit = None
    print("Warning: Visit not available")

try:
    from .image_search import ImageSearch
except ImportError:
    ImageSearch = None
    print("Warning: ImageSearch not available")

try:
    from .zoom import ZoomTool
except ImportError:
    ZoomTool = None
    print("Warning: ZoomTool not available")


__all__ = [
    'BaseTool',
    'TOOL_REGISTRY',
    'register_tool',
    'get_tool',
    'list_tools',
    'get_tool_info',
    'CodeInterpreter',
    'WebSearch',
    'Visit',
    'ImageSearch',
    'ZoomTool',
]

