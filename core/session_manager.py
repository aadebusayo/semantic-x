"""
Universal Session Manager for SemanticX Framework.
Manages the lifecycle of conversation states for each user session.
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
import asyncio

from models.state import ConversationState
from config import settings
from core.memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Manages the lifecycle of ConversationState for each user session.
    
    This acts as the in-memory store for the AI's "cognitive memory" and provides
    session management capabilities including cleanup, timeout handling, and
    session statistics.
    """
    
    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}
        self._session_metadata: Dict[str, Dict] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self.memory = MemoryManager()
        
        # Start cleanup task if not already running
        if not self._cleanup_task or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_expired_sessions())
    
    @classmethod
    def get_instance(cls):
        """Get the singleton instance of SessionManager."""
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def get_or_create_state(self, session_id: str, domain: str = None) -> ConversationState:
        """
        Retrieves the state for a given session ID or creates a new one.
        
        Args:
            session_id: The unique identifier for the user session
            domain: Optional domain context for the session
            
        Returns:
            ConversationState: The state object for the session
        """
        if session_id not in self._sessions:
            logger.info(f"Creating new session state for session_id: {session_id}")
            self._sessions[session_id] = ConversationState(
                session_id=session_id,
                domain=domain
            )
            
            # Initialize session metadata
            self._session_metadata[session_id] = self._build_metadata(domain)
        else:
            logger.info(f"Retrieving existing session state for session_id: {session_id}")
            self._update_metadata(session_id, access_count_increment=1)
            
            # Log the current plan state for debugging
            current_state = self._sessions[session_id]
            if current_state.plan:
                logger.info(f"Session {session_id} has existing plan: {current_state.plan}")
            else:
                logger.info(f"Session {session_id} has no existing plan")
        
        return self._sessions[session_id]

    def save_state(self, state: ConversationState) -> None:
        """
        Saves the updated state back to the session store.
        
        Args:
            state: The ConversationState object to save
        """
        if not state.session_id:
            raise ValueError("ConversationState must have a session_id to be saved.")
        
        logger.debug(f"Saving state for session_id: {state.session_id}")
        
        # Update the state
        self._sessions[state.session_id] = state
        
        self._update_metadata(state.session_id, status=state.status, domain=state.domain)
        
        # Log plan information if available
        if state.plan:
            logger.info(f"Saving state with plan: {state.plan}")
        else:
            logger.info(f"Saving state with no plan")

    def clear_session(self, session_id: str) -> None:
        """
        Removes a session from the store, effectively ending it.
        
        Args:
            session_id: The session to clear
        """
        if session_id in self._sessions:
            logger.info(f"Clearing session state for session_id: {session_id}")
            del self._sessions[session_id]
        
        if session_id in self._session_metadata:
            logger.info(f"Clearing session metadata for session_id: {session_id}")
            del self._session_metadata[session_id]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """
        Get information about a specific session.
        
        Args:
            session_id: The session ID to get info for
            
        Returns:
            Optional[Dict]: Session information or None if not found
        """
        if session_id not in self._sessions:
            return None
        
        state = self._sessions[session_id]
        metadata = self._session_metadata.get(session_id, {})
        
        return {
            "session_id": session_id,
            "conversation_id": state.conversation_id,
            "domain": state.domain,
            "intent": state.intent,
            "sub_intent": state.sub_intent,
            "auth_status": state.auth_status,
            "auth_pending": state.auth_pending,
            "status": state.status,
            "created_at": metadata.get("created_at"),
            "last_activity": metadata.get("last_activity"),
            "access_count": metadata.get("access_count", 0),
            "message_count": len(state.conversation_history),
            "has_plan": bool(state.plan),
            "current_step": state.current_step,
            "current_agent": state.current_agent,
            "routing_metadata": state.routing_metadata,
            "is_task_complete": state.is_task_complete
        }

    def list_active_sessions(self) -> List[Dict]:
        """
        Get a list of all active sessions.
        
        Returns:
            List[Dict]: List of active session information
        """
        active_sessions = []
        
        for session_id in self._sessions:
            session_info = self.get_session_info(session_id)
            if session_info:
                active_sessions.append(session_info)
        
        return active_sessions

    def get_session_count(self) -> int:
        """
        Get the total number of active sessions.
        
        Returns:
            int: Number of active sessions
        """
        return len(self._sessions)

    def is_session_expired(self, session_id: str) -> bool:
        """
        Check if a session has expired based on timeout settings.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            bool: True if session is expired, False otherwise
        """
        if session_id not in self._session_metadata:
            return True
        
        metadata = self._session_metadata[session_id]
        last_activity = metadata.get("last_activity") or metadata.get("created_at") or datetime.now(timezone.utc)
        
        timeout_minutes = settings.session_timeout_minutes
        timeout_delta = timedelta(minutes=timeout_minutes)
        elapsed = datetime.now(timezone.utc) - last_activity
        if elapsed <= timeout_delta:
            return False
        
        state = self._sessions.get(session_id)
        if state and (state.plan or state.auth_pending):
            grace = timedelta(minutes=getattr(settings, "session_plan_grace_minutes", 5))
            if elapsed <= timeout_delta + grace:
                logger.debug("Session %s granted grace due to active plan/auth", session_id)
                return False
        return True

    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions and return the count of cleaned sessions.
        
        Returns:
            int: Number of sessions cleaned up
        """
        expired_sessions = []
        
        for session_id in list(self._sessions.keys()):
            if self.is_session_expired(session_id):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            logger.info(f"Cleaning up expired session: {session_id}")
            self.clear_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        
        return len(expired_sessions)

    async def _cleanup_expired_sessions(self):
        """
        Background task to periodically clean up expired sessions.
        """
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                cleaned_count = self.cleanup_expired_sessions()
                if cleaned_count > 0:
                    logger.info(f"Background cleanup: cleaned {cleaned_count} expired sessions")
            except Exception as e:
                logger.error(f"Error in background cleanup task: {e}")

    def get_session_statistics(self) -> Dict:
        """
        Get statistics about all sessions.
        
        Returns:
            Dict: Session statistics
        """
        total_sessions = len(self._sessions)
        active_sessions = 0
        expired_sessions = 0
        total_messages = 0
        domains = {}
        
        for session_id in self._sessions:
            if self.is_session_expired(session_id):
                expired_sessions += 1
            else:
                active_sessions += 1
            
            state = self._sessions[session_id]
            total_messages += len(state.conversation_history)
            
            domain = state.domain or "unknown"
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "expired_sessions": expired_sessions,
            "total_messages": total_messages,
            "domains": domains,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    def extend_session(self, session_id: str) -> bool:
        """
        Extend the session timeout by updating the last activity.
        
        Args:
            session_id: The session ID to extend
            
        Returns:
            bool: True if session was extended, False if not found
        """
        return self._update_metadata(session_id, touch_only=True)

    def set_session_domain(self, session_id: str, domain: str) -> bool:
        """
        Set or update the domain for a session.
        
        Args:
            session_id: The session ID
            domain: The domain to set
            
        Returns:
            bool: True if domain was set, False if session not found
        """
        if session_id not in self._sessions:
            return False
        
        self._sessions[session_id].domain = domain
        self._update_metadata(session_id, domain=domain)
        
        logger.info(f"Set domain '{domain}' for session_id: {session_id}")
        return True

    def _build_metadata(self, domain: Optional[str]) -> Dict:
        now = datetime.now(timezone.utc)
        return {
            "created_at": now,
            "last_activity": now,
            "access_count": 0,
            "domain": domain,
            "status": "active"
        }

    def _update_metadata(self, session_id: str, status: Optional[str] = None, domain: Optional[str] = None, access_count_increment: int = 0, touch_only: bool = False):
        if session_id not in self._session_metadata:
            self._session_metadata[session_id] = self._build_metadata(domain)
            return True
        metadata = self._session_metadata[session_id]
        if access_count_increment:
            metadata["access_count"] = metadata.get("access_count", 0) + access_count_increment
        if status:
            metadata["status"] = status
        if domain:
            metadata["domain"] = domain
        if touch_only or status or domain or access_count_increment:
            metadata["last_activity"] = datetime.now(timezone.utc)
        return True

    def get_domain_sessions(self, domain: str) -> List[str]:
        """
        Get all session IDs for a specific domain.
        
        Args:
            domain: The domain to filter by
            
        Returns:
            List[str]: List of session IDs in the specified domain
        """
        domain_sessions = []
        
        for session_id, state in self._sessions.items():
            if state.domain == domain:
                domain_sessions.append(session_id)
        
        return domain_sessions

    def shutdown(self):
        """Shutdown the session manager and cleanup resources."""
        logger.info("Shutting down SessionManager")
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        # Clear all sessions
        self._sessions.clear()
        self._session_metadata.clear()
        
        logger.info("SessionManager shutdown complete")


# Global instance
session_manager = SessionManager.get_instance()
