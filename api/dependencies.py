from __future__ import annotations

from fastapi import Request

from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.cosmos_repositories import CosmosChatRepository, CosmosEntityRepository, CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService
from services.llm_service import AzureOpenAILLMService
from services.signalr_service import SignalRService
from services.tts_service import AzureTTSService


def get_search_service(request: Request) -> AzureAISearchService:
    return request.app.state.search_service


def get_chat_repo(request: Request) -> CosmosChatRepository:
    return request.app.state.chat_repo


def get_suggestions_repo(request: Request) -> CosmosSuggestionsRepository:
    return request.app.state.suggestions_repo


def get_entity_repo(request: Request) -> CosmosEntityRepository:
    return request.app.state.entity_repo


def get_document_catalog(request: Request) -> DocumentCatalogService:
    return request.app.state.document_catalog


def get_blob_source(request: Request) -> BlobStorageSource | None:
    return getattr(request.app.state, "blob_source", None)


def get_llm_service(request: Request) -> AzureOpenAILLMService:
    return request.app.state.llm_service


def get_signalr_service(request: Request) -> SignalRService:
    return request.app.state.signalr_service


def get_tts_service(request: Request) -> AzureTTSService:
    return request.app.state.tts_service
