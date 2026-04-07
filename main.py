"""FastAPI entrypoint for Infosearch.

REST-only backend that:
- queries Azure AI Search (semantic + vector)
- returns short summaries with citations
- stores chat history + doc quick-questions in Cosmos DB
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.router import router as api_router
from config import get_settings
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.cosmos_repositories import CosmosChatRepository, CosmosEntityRepository, CosmosIngestionRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService
from services.ingestion_worker import IngestionWorker
from services.llm_service import AzureOpenAILLMService
from services.signalr_service import SignalRService
from services.tts_service import AzureTTSService


load_dotenv()
settings = get_settings()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Reduce Azure SDK HTTP noise (request/response dumps).
for noisy_logger in (
    "azure",
    "azure.core.pipeline.policies.http_logging_policy",
    "azure.cosmos._cosmos_http_logging_policy",
    "azure.storage",
):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Reduce noisy runtime logs.
logging.getLogger("watchfiles.main").setLevel(logging.WARNING)
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create/close shared Azure clients."""
    search_service = AzureAISearchService.from_settings(settings)
    llm_service = AzureOpenAILLMService.from_settings(settings)
    signalr_service = SignalRService.from_settings(settings)
    tts_service = AzureTTSService.from_settings(settings)
    chat_repo = CosmosChatRepository.from_settings(settings)
    suggestions_repo = CosmosSuggestionsRepository.from_settings(settings)
    ingestion_repo = CosmosIngestionRepository.from_settings(settings)
    entity_repo = CosmosEntityRepository.from_settings(settings)

    ingestion_worker = None
    blob_source = None
    document_catalog = None

    async def _safe_close(name: str, closer):
        try:
            await closer()
        except BaseException as exc:
            logger.warning("Error closing %s: %s", name, exc)

    app.state.search_service = search_service
    app.state.llm_service = llm_service
    app.state.signalr_service = signalr_service
    app.state.tts_service = tts_service
    app.state.chat_repo = chat_repo
    app.state.suggestions_repo = suggestions_repo
    app.state.ingestion_repo = ingestion_repo
    app.state.entity_repo = entity_repo
    app.state.blob_source = None
    app.state.document_catalog = None

    try:
        await llm_service.open()
        await signalr_service.open()
        await chat_repo.open()
        await suggestions_repo.open()
        await ingestion_repo.open()
        await entity_repo.open()

        if settings.azure_storage_connection_string and settings.azure_storage_container:
            blob_source = BlobStorageSource(
                connection_string=settings.azure_storage_connection_string,
                container=settings.azure_storage_container,
                prefix=settings.azure_storage_prefix,
            )
            await blob_source.open()
            app.state.blob_source = blob_source

        document_catalog = DocumentCatalogService(
            ingestion_repo=ingestion_repo,
            entity_repo=entity_repo,
            blob_source=blob_source,
        )
        app.state.document_catalog = document_catalog

        # Optional: pull-based ingestion (no messaging/queue required)
        if (
            settings.enable_ingestion_worker
            and blob_source is not None
        ):
            ingestion_worker = IngestionWorker(
                settings=settings,
                blob_source=blob_source,
                search_service=search_service,
                ingestion_repo=ingestion_repo,
                suggestions_repo=suggestions_repo,
                document_catalog=document_catalog,
            )
            try:
                await ingestion_worker.open()
                ingestion_worker.start()
                app.state.ingestion_worker = ingestion_worker
                logger.info("Ingestion worker started (poll=%ss)", settings.ingestion_poll_seconds)
            except (ModuleNotFoundError, ImportError) as exc:
                ingestion_worker = None
                logger.warning("Ingestion worker disabled due to missing async transport dependency: %s", exc)

        logger.info("Infosearch API started")
        yield
    finally:
        if ingestion_worker is not None:
            await _safe_close("ingestion worker stop", ingestion_worker.stop)
            await _safe_close("ingestion worker", ingestion_worker.close)

        if blob_source is not None:
            await _safe_close("blob source", blob_source.close)

        await _safe_close("chat repository", chat_repo.close)
        await _safe_close("suggestions repository", suggestions_repo.close)
        await _safe_close("ingestion repository", ingestion_repo.close)
        await _safe_close("entity repository", entity_repo.close)
        await _safe_close("llm service", llm_service.close)
        await _safe_close("signalr service", signalr_service.close)
        await _safe_close("search service", search_service.close)
        logger.info("Infosearch API stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Infosearch REST API",
    debug=settings.debug,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/frontend", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/playground", include_in_schema=False)
async def playground():
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        return JSONResponse(status_code=404, content={"error": "frontend_not_found"})
    return FileResponse(index)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


app.include_router(api_router, prefix="/api/v1")


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level="info",
    )
