"""
Core package for SemanticX Framework.
Contains the fundamental components for agent orchestration and management.
"""

from .base_agent import BaseAgent
from .tool_registry import ToolRegistry

# Lazy imports to avoid circular dependencies during testing
def __getattr__(name: str):
    """Lazy import for heavy modules that may have circular dependencies."""
    if name == "Orchestrator":
        from .orchestrator import Orchestrator
        return Orchestrator
    elif name == "SessionManager":
        from .session_manager import SessionManager
        return SessionManager
    elif name == "MemoryManager":
        from .memory_manager import MemoryManager
        return MemoryManager
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "BaseAgent",
    "Orchestrator",
    "SessionManager",
    "MemoryManager",
    "ToolRegistry"
]
