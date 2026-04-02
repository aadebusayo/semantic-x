"""Infosearch API configuration.

This service is intentionally lean:
- Query Azure AI Search (semantic + vector) for relevant document chunks
- Return short, extractive summaries with citations metadata
- Persist chat history and document quick-questions in Azure Cosmos DB

No other external API calls are required.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # App
    app_name: str = Field(default="Infosearch API")
    app_version: str = Field(default="0.1.0")
    debug: bool = Field(default=False)

    # Server
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000)
    reload: bool = Field(default=True)

    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_credentials: bool = Field(default=True)

    # Azure AI Search
    azure_search_endpoint: Optional[str] = Field(default=None, description="https://<service>.search.windows.net")
    azure_search_index: Optional[str] = Field(default=None)
    azure_search_api_key: Optional[str] = Field(default=None)
    azure_search_semantic_config: Optional[str] = Field(default=None)

    search_content_field: str = Field(default="content")
    search_vector_field: str = Field(default="contentVector")
    search_vector_dimensions: int = Field(default=1536)
    search_vector_profile_vectorizer: str = Field(default="default-aoai-vectorizer")
    search_doc_id_field: str = Field(default="documentId")
    search_doc_name_field: str = Field(default="documentName")

    # Search indexing (chunk documents)
    search_chunk_key_field: str = Field(default="id", description="Key field name in the chunk index")
    search_chunk_index_field: str = Field(default="chunkIndex")
    search_chunk_start_field: str = Field(default="chunkStart")
    search_chunk_end_field: str = Field(default="chunkEnd")

    # If your index supports filtering by doc id/name, set these to the filterable field names.
    search_filter_doc_id_field: Optional[str] = Field(default=None)
    search_filter_doc_name_field: Optional[str] = Field(default=None)

    default_top_k: int = Field(default=5)
    max_excerpt_chars: int = Field(default=280)
    # Minimum @search.score for a citation to be included; 0.0 = no filter.
    search_min_citation_score: float = Field(default=0.02)

    # Azure OpenAI (chat generation)
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_api_key: Optional[str] = Field(default=None)
    azure_openai_api_version: str = Field(default="2024-02-15-preview")
    azure_openai_chat_deployment: Optional[str] = Field(default=None)
    azure_openai_embedding_deployment: Optional[str] = Field(default=None)
    azure_openai_embedding_model: str = Field(default="text-embedding-3-small")
    azure_openai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    azure_openai_max_tokens: int = Field(default=4096, ge=32, le=16000)

    # Streaming / SignalR
    signalr_enabled: bool = Field(default=False)
    signalr_endpoint: Optional[str] = Field(default=None)
    signalr_access_token: Optional[str] = Field(default=None)
    signalr_hub: str = Field(default="chat")
    signalr_target: str = Field(default="chatStream")

    # Cosmos DB
    cosmos_endpoint: Optional[str] = Field(default=None)
    cosmos_key: Optional[str] = Field(default=None)
    cosmos_database: str = Field(default="infosearch")
    cosmos_chat_container: str = Field(default="chats")
    cosmos_suggestions_container: str = Field(default="document_suggestions")
    cosmos_ingestion_container: str = Field(default="ingestion")
    cosmos_entity_container: str = Field(default="Entity")

    entity_root_basepath_id: str = Field(default="0000-0000-0000-0000")
    entity_file_object_type: int = Field(default=0)

    # Azure Blob Storage (source documents)
    azure_storage_connection_string: Optional[str] = Field(default=None)
    azure_storage_container: Optional[str] = Field(default=None)
    azure_storage_prefix: Optional[str] = Field(default=None, description="Optional blob name prefix")

    enable_ingestion_worker: bool = Field(default=True)

    ingestion_poll_seconds: int = Field(default=60, ge=5, le=3600)
    ingestion_reconcile_deletions: bool = Field(default=True)
    ingestion_reconcile_every_polls: int = Field(default=10, ge=1, le=1000)

    # Indexing behavior
    chunk_size_chars: int = Field(default=1200, ge=200, le=5000)
    chunk_overlap_chars: int = Field(default=150, ge=0, le=1000)

    # Azure Speech (Text-to-Speech)
    azure_speech_key: Optional[str] = Field(default=None)
    azure_speech_region: Optional[str] = Field(default=None)
    azure_speech_voice: str = Field(default="en-US-AvaMultilingualNeural")

    # Chat behavior
    max_chat_messages: int = Field(default=60)
    default_user_id: str = Field(default="anonymous")


@lru_cache
def get_settings() -> Settings:
    return Settings()
