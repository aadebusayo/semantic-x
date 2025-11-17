"""
Main entry point for SemanticX Framework.
Provides a universal AI agent orchestration service.
"""
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from config import settings, validate_config
from core.session_manager import session_manager
from services.tool_handler import ToolHandler
from core.tool_registry import ToolRegistry
from utils.prompt_utils import PromptManager
from api.router import router as websocket_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format,
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)
# Basic metrics
REQUEST_COUNT = Counter("semanticx_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
REQUEST_LATENCY = Histogram("semanticx_request_latency_seconds", "Request latency", ["endpoint"]) 

# Suppress noisy logs
logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    try:
        logger.info(" Starting SemanticX Framework...")
        
        # Soft-validate configuration (do not block startup)
        try:
            validate_config()
        except Exception as _:
            logger.warning("Configuration validation reported issues, continuing with defaults for dev mode.")
        
        # Initialize core services
        logger.info("🔧 Initializing core services...")
        
        # Initialize tool registry (loads schemas at startup)
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        logger.info(f" Loaded {len(tools)} tools for function calling")
        
        # Initialize prompt manager
        prompt_manager = PromptManager()
        logger.info(" Prompt manager initialized")
        
        # Initialize session manager
        logger.info(" Session manager initialized")
        
        logger.info(f" SemanticX Framework ready on {settings.host}:{settings.port}")
        
    except Exception as e:
        logger.error(f" Startup error: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info(" Shutting down SemanticX Framework...")
        
        # Shutdown session manager
        session_manager.shutdown()
        
        logger.info("SemanticX Framework shutdown complete")
        
    except Exception as e:
        logger.error(f" Shutdown error: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="Universal AI agent orchestration framework for building intelligent conversational AI systems",
    version=settings.app_version,
    debug=settings.debug
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set lifespan
app.lifespan = lifespan


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with framework information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Universal AI agent orchestration framework",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs",
        "health": "/health"
    }
@app.get("/metrics")
async def metrics():
    content = generate_latest()
    return JSONResponse(content=content, media_type=CONTENT_TYPE_LATEST)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check session manager
        session_count = session_manager.get_session_count()
        session_stats = session_manager.get_session_statistics()
        
        # Check tool handler
        registry = ToolRegistry()
        tools_count = len(registry.get_all_tools())
        
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": settings.app_version,
            "components": {
                "session_manager": "healthy",
                "tool_handler": "healthy",
                "prompt_manager": "healthy"
            },
            "metrics": {
                "active_sessions": session_count,
                "total_tools": tools_count,
                "uptime": "running"
            },
            "session_statistics": session_stats
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Framework information endpoint
@app.get("/info")
async def framework_info():
    """Get detailed framework information."""
    try:
        tool_handler = ToolHandler()
        prompt_manager = PromptManager()
        
        return {
            "framework": {
                "name": settings.app_name,
                "version": settings.app_version,
                "description": "Universal AI agent orchestration framework"
            },
            "configuration": {
                "llm_provider": settings.llm_provider,
                "vector_store_type": settings.vector_store_type,
                "domain": "universal",
                "debug_mode": settings.debug
            },
            "capabilities": {
                "agents": "pluggable_agent_system",
                "orchestration": "multi_step_workflow_management",
                "tools": "dynamic_api_integration",
                "memory": "conversation_memory_and_vector_storage",
                "error_handling": "intelligent_error_analysis",
                "real_time": "websocket_communication"
            },
            "services": {
                "tools_loaded": len(tool_handler.get_all_tools()),
                "prompts_available": prompt_manager.get_available_prompts(),
                "sessions_active": session_manager.get_session_count()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Framework info failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get framework info: {str(e)}")


# Session management endpoints
@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    try:
        sessions = session_manager.list_active_sessions()
        return {
            "sessions": sessions,
            "total_count": len(sessions),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    try:
        session_info = session_manager.get_session_info(session_id)
        if not session_info:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session": session_info,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session info: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    try:
        session_manager.clear_session(session_id)
        return {
            "message": f"Session {session_id} deleted successfully",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


# Tool management endpoints
@app.get("/tools")
async def list_tools():
    """List all available tools."""
    try:
        registry = ToolRegistry()
        tools = registry.get_all_tools()
        
        return {
            "tools": tools,
            "total_count": len(tools),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list tools: {str(e)}")


@app.get("/tools/{agent_name}")
async def get_tools_for_agent(agent_name: str):
    """Get tools available for a specific agent."""
    try:
        tool_handler = ToolHandler()
        tools = tool_handler.get_tools_for_agent(agent_name)
        
        return {
            "agent": agent_name,
            "tools": tools,
            "tool_count": len(tools),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get tools for agent: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get tools for agent: {str(e)}")


# Prompt management endpoints
@app.get("/prompts")
async def list_prompts():
    """List all available prompts."""
    try:
        prompt_manager = PromptManager()
        prompts = prompt_manager.get_available_prompts()
        
        return {
            "prompts": prompts,
            "total_count": len(prompts),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list prompts: {str(e)}")


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )


# Include API/WebSocket router
app.include_router(websocket_router, prefix="/api/v1")


if __name__ == "__main__":
    logger.info(f"Starting SemanticX Framework on {settings.host}:{settings.port}")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
