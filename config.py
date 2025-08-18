"""
Configuration management for SemanticX Framework.
Provides universal configuration that can be extended for any domain.
"""
import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class SemanticXSettings(BaseSettings):
    """Universal configuration settings for SemanticX Framework."""
    
    # Application Settings
    app_name: str = Field(default="SemanticX Framework", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server Settings
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=True, description="Auto-reload on changes")
    
    # LLM Provider Settings
    llm_provider: str = Field(default="openai", description="LLM provider (openai, azure, anthropic, etc.)")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_base_url: Optional[str] = Field(default=None, description="OpenAI base URL")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature")
    openai_max_tokens: int = Field(default=1000, description="OpenAI max tokens")
    
    # Azure OpenAI Settings (alternative to OpenAI)
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_deployment_name: Optional[str] = Field(default=None, description="Azure OpenAI deployment name")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", description="Azure OpenAI API version")
    
    # Vector Database Settings
    vector_store_type: str = Field(default="memory", description="Vector store type (memory, pinecone, weaviate, chroma, qdrant)")
    vector_store_url: Optional[str] = Field(default=None, description="Vector store URL")
    vector_store_api_key: Optional[str] = Field(default=None, description="Vector store API key")
    vector_store_dimension: int = Field(default=1536, description="Vector embedding dimension")
    
    # Pinecone Settings
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index_name: Optional[str] = Field(default=None, description="Pinecone index name")
    
    # Weaviate Settings
    weaviate_url: Optional[str] = Field(default=None, description="Weaviate URL")
    weaviate_api_key: Optional[str] = Field(default=None, description="Weaviate API key")
    
    # Chroma Settings
    chroma_persist_directory: str = Field(default="./chroma_db", description="Chroma persist directory")
    
    # Qdrant Settings
    qdrant_url: Optional[str] = Field(default=None, description="Qdrant URL")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key")
    qdrant_collection: str = Field(default="semanticx", description="Qdrant collection name")
    
    # Tool and Schema Settings
    tool_schema_dir: str = Field(default="./schemas", description="Directory containing OpenAPI schemas")
    tool_defaults_file: str = Field(default="./tool_defaults.json", description="Tool defaults configuration file")
    
    # Prompt Settings
    prompt_dir: str = Field(default="./prompts", description="Directory containing prompt templates")
    
    # Session Settings
    session_timeout_minutes: int = Field(default=30, description="Session timeout in minutes")
    max_conversation_history: int = Field(default=100, description="Maximum conversation history length")
    
    # Memory Settings
    enable_conversation_summarization: bool = Field(default=True, description="Enable conversation summarization")
    enable_vector_storage: bool = Field(default=True, description="Enable vector storage for conversations")
    memory_retention_days: int = Field(default=90, description="Memory retention period in days")
    
    # Error Handling Settings
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts for operations")
    retry_delay_seconds: float = Field(default=1.0, description="Base retry delay in seconds")
    
    # Logging Settings
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", description="Log format")
    
    # CORS Settings
    cors_origins: list = Field(default=["*"], description="CORS allowed origins")
    cors_credentials: bool = Field(default=True, description="CORS allow credentials")
    
    # Security Settings
    enable_rate_limiting: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")
    rate_limit_window: int = Field(default=60, description="Rate limit window in seconds")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = SemanticXSettings()


def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration based on provider."""
    if settings.llm_provider.lower() == "azure":
        return {
            "provider": "azure",
            "api_key": settings.azure_openai_api_key,
            "endpoint": settings.azure_openai_endpoint,
            "deployment_name": settings.azure_openai_deployment_name,
            "api_version": settings.azure_openai_api_version,
        }
    else:  # Default to OpenAI
        return {
            "provider": "openai",
            "api_key": settings.openai_api_key,
            "base_url": settings.openai_base_url,
            "model": settings.openai_model,
            "temperature": settings.openai_temperature,
            "max_tokens": settings.openai_max_tokens,
        }


def get_vector_store_config() -> Dict[str, Any]:
    """Get vector store configuration based on type."""
    config = {
        "type": settings.vector_store_type,
        "dimension": settings.vector_store_dimension,
    }
    
    if settings.vector_store_type == "pinecone":
        config.update({
            "api_key": settings.pinecone_api_key,
            "environment": settings.pinecone_environment,
            "index_name": settings.pinecone_index_name,
        })
    elif settings.vector_store_type == "weaviate":
        config.update({
            "url": settings.weaviate_url,
            "api_key": settings.weaviate_api_key,
        })
    elif settings.vector_store_type == "chroma":
        config.update({
            "persist_directory": settings.chroma_persist_directory,
        })
    elif settings.vector_store_type == "qdrant":
        config.update({
            "url": settings.qdrant_url,
            "api_key": settings.qdrant_api_key,
            "collection": settings.qdrant_collection,
        })
    
    return config


def validate_config() -> bool:
    """Validate that required configuration is present."""
    errors = []
    
    # Check LLM configuration
    llm_config = get_llm_config()
    if llm_config["provider"] == "openai" and not llm_config["api_key"]:
        errors.append("OpenAI API key is required")
    elif llm_config["provider"] == "azure" and not llm_config["api_key"]:
        errors.append("Azure OpenAI API key is required")
    
    # Check vector store configuration
    vector_config = get_vector_store_config()
    if vector_config["type"] != "memory":
        if vector_config["type"] == "pinecone" and not vector_config.get("api_key"):
            errors.append("Pinecone API key is required")
        elif vector_config["type"] == "weaviate" and not vector_config.get("url"):
            errors.append("Weaviate URL is required")
        elif vector_config["type"] == "qdrant" and not vector_config.get("url"):
            errors.append("Qdrant URL is required")
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True
