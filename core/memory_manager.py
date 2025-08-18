"""
Memory Manager for SemanticX Framework.
Provides simple in-memory conversation summarization and vector hooks.
"""
import logging
from typing import List, Dict, Any, Optional

from ..config import settings, get_vector_store_config

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Minimal memory manager with in-memory storage and vector hooks.
    """
    
    def __init__(self):
        self.enabled = settings.enable_conversation_summarization or settings.enable_vector_storage
        self.vector_config = get_vector_store_config()
        self.store: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("MemoryManager initialized")
    
    def add_memory(self, session_id: str, item: Dict[str, Any]):
        self.store.setdefault(session_id, []).append(item)
    
    def get_memories(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        items = self.store.get(session_id, [])
        return items[-limit:]
    
    def clear_session(self, session_id: str):
        self.store.pop(session_id, None)
    
    def summarize(self, session_id: str) -> Optional[str]:
        # Placeholder for real summarization via LLM
        messages = self.store.get(session_id, [])
        if not messages:
            return None
        return f"Summary of {len(messages)} memory items."
