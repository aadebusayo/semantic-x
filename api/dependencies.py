from __future__ import annotations

from fastapi import Request

from services.azure_ai_search import AzureAISearchService
from services.cosmos_repositories import CosmosChatRepository, CosmosSuggestionsRepository
from services.llm_service import AzureOpenAILLMService
from services.signalr_service import SignalRService
from services.tts_service import AzureTTSService


def get_search_service(request: Request) -> AzureAISearchService:
    return request.app.state.search_service


def get_chat_repo(request: Request) -> CosmosChatRepository:
    return request.app.state.chat_repo


def get_suggestions_repo(request: Request) -> CosmosSuggestionsRepository:
    return request.app.state.suggestions_repo


def get_llm_service(request: Request) -> AzureOpenAILLMService:
    return request.app.state.llm_service


def get_signalr_service(request: Request) -> SignalRService:
    return request.app.state.signalr_service


def get_tts_service(request: Request) -> AzureTTSService:
    return request.app.state.tts_service
