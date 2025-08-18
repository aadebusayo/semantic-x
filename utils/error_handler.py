"""
Intelligent Error Handler for SemanticX Framework.
Provides LLM-driven error analysis and user-friendly error messages.
"""
import logging
from typing import Dict, Any, Optional
import traceback

logger = logging.getLogger(__name__)


class IntelligentErrorHandler:
    """
    Handles errors intelligently and provides user-friendly messages.
    """
    
    def __init__(self):
        self.error_patterns = {
            "llm_call": "I'm having trouble processing your request right now.",
            "tool_execution": "There was an issue with one of the tools I was trying to use.",
            "validation_error": "I received some information that wasn't quite right.",
            "network_error": "I'm having trouble connecting to external services.",
            "timeout_error": "The operation is taking longer than expected.",
            "permission_error": "I don't have permission to perform that action.",
            "not_found_error": "I couldn't find what you were looking for."
        }
        
        self.retryable_errors = {
            "llm_call": True,
            "tool_execution": True,
            "validation_error": True,
            "network_error": True,
            "timeout_error": True,
            "permission_error": False,
            "not_found_error": False
        }
        
        logger.info("IntelligentErrorHandler initialized")
    
    async def get_user_friendly_error_message(self, error: Exception, error_type: str = "general") -> str:
        """
        Get a user-friendly error message for the given error.
        
        Args:
            error: The exception that occurred
            error_type: Type of error for categorization
            
        Returns:
            User-friendly error message
        """
        try:
            # Get base message for error type
            base_message = self.error_patterns.get(error_type, self.error_patterns["general"])
            
            # Analyze the error for specific details
            error_details = self._analyze_error(error)
            
            # Combine base message with specific details
            if error_details:
                message = f"{base_message} {error_details}"
            else:
                message = base_message
            
            # Add retry suggestion if applicable
            if self.retryable_errors.get(error_type, False):
                message += " Please try again in a moment."
            
            return message
            
        except Exception as e:
            logger.error(f"Error generating user-friendly message: {e}")
            return "I encountered an unexpected error. Please try again."
    
    def _analyze_error(self, error: Exception) -> Optional[str]:
        """
        Analyze an error to extract meaningful details.
        
        Args:
            error: The exception to analyze
            
        Returns:
            Optional string with error details
        """
        try:
            error_str = str(error).lower()
            
            # Check for common error patterns
            if "timeout" in error_str or "timed out" in error_str:
                return "The operation timed out."
            elif "network" in error_str or "connection" in error_str:
                return "There's a network connectivity issue."
            elif "permission" in error_str or "unauthorized" in error_str:
                return "I don't have the right permissions for this."
            elif "not found" in error_str or "404" in error_str:
                return "The requested resource wasn't found."
            elif "validation" in error_str or "invalid" in error_str:
                return "The input data wasn't in the expected format."
            elif "rate limit" in error_str or "too many requests" in error_str:
                return "I'm making too many requests too quickly."
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error analyzing error: {e}")
            return None
    
    def is_retryable(self, error: Exception, error_type: str = "general") -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception that occurred
            error_type: Type of error for categorization
            
        Returns:
            True if the error is retryable
        """
        return self.retryable_errors.get(error_type, False)
    
    def get_error_context(self, error: Exception) -> Dict[str, Any]:
        """
        Get context information about an error.
        
        Args:
            error: The exception that occurred
            
        Returns:
            Dict containing error context
        """
        return {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "retryable": self.is_retryable(error)
        }
    
    def log_error(self, error: Exception, context: str = "", error_type: str = "general"):
        """
        Log an error with context.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            error_type: Type of error for categorization
        """
        error_context = self.get_error_context(error)
        logger.error(
            f"Error in {context}: {error_context['error_message']}",
            extra={
                "error_type": error_type,
                "error_context": error_context,
                "context": context
            }
        )
