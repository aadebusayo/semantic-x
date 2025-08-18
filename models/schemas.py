"""
API schemas for SemanticX Framework.
Defines request and response models for the API endpoints.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum


class RequestType(str, Enum):
    """Types of requests that can be made."""
    TEXT = "text"
    VOICE = "voice"
    AUTH = "auth"
    COMMAND = "command"


class ResponseType(str, Enum):
    """Types of responses that can be sent."""
    TEXT = "text"
    VOICE = "voice"
    ERROR = "error"
    ACK = "ack"
    STATUS = "status"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""
    type: str = Field(..., description="Message type")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z", description="Message timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")


class Request(WebSocketMessage):
    """Request message from client."""
    type: str = Field("request", description="Message type")
    message: str = Field(..., description="User message content")
    request_type: RequestType = Field(RequestType.TEXT, description="Type of request")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional request metadata")


class Response(WebSocketMessage):
    """Response message to client."""
    type: ResponseType = Field(..., description="Response type")
    message: str = Field(..., description="Response message content")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional response metadata")
    
    def filtered_dict(self) -> Dict[str, Any]:
        """Return a filtered dictionary for serialization."""
        return {
            "type": self.type.value if hasattr(self.type, 'value') else self.type,
            "message": self.message,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "metadata": self.metadata
        }


class ErrorResponse(Response):
    """Error response message."""
    type: ResponseType = Field(ResponseType.ERROR, description="Response type")
    error_code: Optional[str] = Field(None, description="Error code")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Detailed error information")
    suggested_action: Optional[str] = Field(None, description="Suggested action to resolve the error")


class StatusResponse(Response):
    """Status response message."""
    type: ResponseType = Field(ResponseType.STATUS, description="Response type")
    status: str = Field(..., description="Current status")
    progress: Optional[float] = Field(None, description="Progress percentage (0-100)")
    estimated_completion: Optional[str] = Field(None, description="Estimated completion time")


class ToolCallRequest(WebSocketMessage):
    """Tool call request message."""
    type: str = Field("tool_call", description="Message type")
    function_name: str = Field(..., description="Name of the function to call")
    arguments: Dict[str, Any] = Field(..., description="Function arguments")
    tool_call_id: str = Field(..., description="Unique identifier for the tool call")


class ToolResultResponse(WebSocketMessage):
    """Tool result response message."""
    type: ResponseType = Field(ResponseType.TOOL_RESULT, description="Response type")
    tool_call_id: str = Field(..., description="ID of the tool call this result corresponds to")
    function_name: str = Field(..., description="Name of the function that was called")
    result: Any = Field(..., description="Result of the tool call")
    success: bool = Field(True, description="Whether the tool call was successful")
    error_message: Optional[str] = Field(None, description="Error message if the call failed")


class AuthenticationRequest(WebSocketMessage):
    """Authentication request message."""
    type: str = Field("auth", description="Message type")
    credentials: Dict[str, Any] = Field(..., description="Authentication credentials")
    auth_method: str = Field(..., description="Authentication method (token, password, etc.)")


class AuthenticationResponse(WebSocketMessage):
    """Authentication response message."""
    type: str = Field("auth_response", description="Message type")
    success: bool = Field(..., description="Whether authentication was successful")
    token: Optional[str] = Field(None, description="Authentication token if successful")
    user_profile: Optional[Dict[str, Any]] = Field(None, description="User profile information")
    error_message: Optional[str] = Field(None, description="Error message if authentication failed")


class ConversationSummary(BaseModel):
    """Summary of a conversation for memory storage."""
    conversation_id: str = Field(..., description="Conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    summary: str = Field(..., description="Summarized conversation content")
    intent: Optional[str] = Field(None, description="Primary intent of the conversation")
    sub_intent: Optional[str] = Field(None, description="Sub-intent of the conversation")
    domain: Optional[str] = Field(None, description="Domain context")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Creation timestamp")
    vector_embedding: Optional[List[float]] = Field(None, description="Vector representation")


class MemoryQuery(BaseModel):
    """Query for searching conversation memory."""
    query: str = Field(..., description="Search query")
    domain: Optional[str] = Field(None, description="Domain to search in")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: int = Field(default=10, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score")


class MemorySearchResult(BaseModel):
    """Result of a memory search."""
    conversation_id: str = Field(..., description="Conversation identifier")
    summary: str = Field(..., description="Conversation summary")
    similarity_score: float = Field(..., description="Similarity score")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    created_at: str = Field(..., description="When the conversation occurred")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat(), description="Check timestamp")
    version: str = Field(..., description="Framework version")
    components: Dict[str, str] = Field(default_factory=dict, description="Component statuses")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")


class AgentInfo(BaseModel):
    """Information about an available agent."""
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    domain: str = Field(..., description="Domain the agent specializes in")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    available_tools: List[str] = Field(default_factory=list, description="Tools available to this agent")
    is_active: bool = Field(True, description="Whether the agent is currently active")


class ToolInfo(BaseModel):
    """Information about an available tool."""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    category: str = Field(..., description="Tool category")
    parameters: Dict[str, Any] = Field(..., description="Tool parameters schema")
    required_parameters: List[str] = Field(default_factory=list, description="Required parameters")
    is_active: bool = Field(True, description="Whether the tool is currently active")
