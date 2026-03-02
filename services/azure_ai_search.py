from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizableTextQuery

from config import Settings
from models.infosearch_api import Citation, DocumentRef

logger = logging.getLogger(__name__)


def _escape_odata_string(value: str) -> str:
    # OData uses single quotes; escape by doubling.
    return value.replace("'", "''")


class AzureAISearchService:
    def __init__(
        self,
        *,
        endpoint: Optional[str],
        index_name: Optional[str],
        api_key: Optional[str],
        semantic_config: Optional[str],
        content_field: str,
        vector_field: str,
        doc_id_field: str,
        doc_name_field: str,
        filter_doc_id_field: Optional[str],
        filter_doc_name_field: Optional[str],
        max_excerpt_chars: int,
        min_citation_score: float = 0.0,
    ):
        self._endpoint = endpoint
        self._index_name = index_name
        self._api_key = api_key
        self._semantic_config = semantic_config
        self._content_field = content_field
        self._vector_field = vector_field
        self._doc_id_field = doc_id_field
        self._doc_name_field = doc_name_field
        self._filter_doc_id_field = filter_doc_id_field
        self._filter_doc_name_field = filter_doc_name_field
        self._max_excerpt_chars = max_excerpt_chars
        self._min_citation_score = min_citation_score

        self._client: Optional[SearchClient] = None
        self._chunk_key_field: str = "id"
        self._chunk_index_field: str = "chunkIndex"
        self._chunk_start_field: str = "chunkStart"
        self._chunk_end_field: str = "chunkEnd"
        self._vector_query_enabled: bool = True

    @classmethod
    def from_settings(cls, settings: Settings) -> "AzureAISearchService":
        inst = cls(
            endpoint=settings.azure_search_endpoint,
            index_name=settings.azure_search_index,
            api_key=settings.azure_search_api_key,
            semantic_config=settings.azure_search_semantic_config,
            content_field=settings.search_content_field,
            vector_field=settings.search_vector_field,
            doc_id_field=settings.search_doc_id_field,
            doc_name_field=settings.search_doc_name_field,
            filter_doc_id_field=settings.search_filter_doc_id_field,
            filter_doc_name_field=settings.search_filter_doc_name_field,
            max_excerpt_chars=settings.max_excerpt_chars,
            min_citation_score=settings.search_min_citation_score,
        )

        inst._chunk_key_field = settings.search_chunk_key_field
        inst._chunk_index_field = settings.search_chunk_index_field
        inst._chunk_start_field = settings.search_chunk_start_field
        inst._chunk_end_field = settings.search_chunk_end_field
        return inst

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _get_client(self) -> Optional[SearchClient]:
        if not self._endpoint or not self._index_name or not self._api_key:
            return None
        if self._client is None:
            self._client = SearchClient(
                endpoint=self._endpoint,
                index_name=self._index_name,
                credential=AzureKeyCredential(self._api_key),
            )
        return self._client

    def _build_filter(self, document: Optional[DocumentRef]) -> Optional[str]:
        if not document:
            return None

        if document.id and self._filter_doc_id_field:
            return f"{self._filter_doc_id_field} eq '{_escape_odata_string(document.id)}'"

        if document.name and self._filter_doc_name_field:
            return f"{self._filter_doc_name_field} eq '{_escape_odata_string(document.name)}'"

        # No filterable field configured
        return None

    def _make_excerpt(self, result: Dict[str, Any]) -> Optional[str]:
        highlights = result.get("@search.highlights") or {}
        hl = highlights.get(self._content_field)
        if isinstance(hl, list) and hl:
            excerpt = " ... ".join([h.strip() for h in hl[:2] if isinstance(h, str)])
            return excerpt[: self._max_excerpt_chars]

        content = result.get(self._content_field)
        if isinstance(content, str) and content.strip():
            return content.strip()[: self._max_excerpt_chars]

        return None

    @staticmethod
    def _is_vector_profile_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "invalidvectorquery" in text
            or "vectorizer defined" in text
            or "parameter name: vector.fields" in text
        )

    async def search(self, *, query: str, document: Optional[DocumentRef], top_k: int) -> List[Citation]:
        client = self._get_client()
        if client is None:
            logger.warning("Azure AI Search is not configured; returning empty citations")
            return []

        filter_expr = self._build_filter(document)

        select_fields = list(
            {
                self._content_field,
                self._doc_id_field,
                self._doc_name_field,
            }
        )

        query_type = "semantic" if self._semantic_config else "simple"

        search_kwargs: Dict[str, Any] = {
            "search_text": query,
            "top": top_k,
            "query_type": query_type,
            "semantic_configuration_name": self._semantic_config,
            "select": select_fields,
            "highlight_fields": self._content_field,
            "filter": filter_expr,
        }

        async def _collect(pageable) -> List[Citation]:
            citations: List[Citation] = []
            idx = 0
            async for item in pageable:
                idx += 1
                result: Dict[str, Any] = dict(item)
                score = result.get("@search.score")
                if self._min_citation_score > 0 and (score is None or score < self._min_citation_score):
                    logger.debug("Skipping citation with score=%s (below threshold %s)", score, self._min_citation_score)
                    continue
                citations.append(
                    Citation(
                        document_name=result.get(self._doc_name_field),
                        document_id=result.get(self._doc_id_field),
                        rank=idx,
                        score=score,
                        excerpt=self._make_excerpt(result),
                    )
                )
            return citations

        # First try hybrid/vector search (if enabled). If it fails, fall back to plain search.
        if self._vector_query_enabled and self._vector_field:
            try:
                vector_results = await client.search(
                    **search_kwargs,
                    vector_queries=[
                        VectorizableTextQuery(
                            text=query,
                            k_nearest_neighbors=top_k,
                            fields=self._vector_field,
                        )
                    ],
                )
                return await _collect(vector_results)
            except Exception as e:
                if self._is_vector_profile_error(e):
                    self._vector_query_enabled = False
                    logger.warning(
                        "Vector query disabled for this runtime due to index vector profile mismatch; using plain search fallback."
                    )
                else:
                    logger.warning("Vector query failed; retrying without vector query: %s", e)

        try:
            plain_results = await client.search(**search_kwargs)
            return await _collect(plain_results)
        except Exception as e:
            logger.error("Search failed after vector fallback: %s", e, exc_info=True)
            return []

    async def upsert_chunks(
        self,
        *,
        document_id: str,
        document_name: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Upsert chunk documents into Azure AI Search.

        Each chunk dict should have: id, content, chunkIndex, chunkStart, chunkEnd.
        """
        client = self._get_client()
        if client is None:
            logger.warning("Azure AI Search is not configured; skipping indexing")
            return

        docs: List[Dict[str, Any]] = []
        for ch in chunks:
            docs.append(
                {
                    self._chunk_key_field: ch["id"],
                    self._doc_id_field: document_id,
                    self._doc_name_field: document_name,
                    self._content_field: ch.get("content") or "",
                    self._chunk_index_field: int(ch.get("chunkIndex", 0)),
                    self._chunk_start_field: int(ch.get("chunkStart", 0)),
                    self._chunk_end_field: int(ch.get("chunkEnd", 0)),
                }
            )

        # Batch to avoid request limits.
        batch_size = 200
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            await client.merge_or_upload_documents(documents=batch)
