from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

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

        filters: List[str] = []

        if document.id and self._filter_doc_id_field:
            filters.append(f"{self._filter_doc_id_field} eq '{_escape_odata_string(document.id)}'")

        if document.name and self._filter_doc_name_field:
            filters.append(f"{self._filter_doc_name_field} eq '{_escape_odata_string(document.name)}'")

        if filters:
            if len(filters) == 1:
                return filters[0]
            return " or ".join(f"({filter_expr})" for filter_expr in filters)

        # No filterable field configured
        return None

    def _document_filter_field(self) -> Optional[str]:
        return self._filter_doc_id_field or self._doc_id_field

    @staticmethod
    def _normalize_match_value(value: Optional[str]) -> Optional[str]:
        text = (value or "").strip()
        return text.casefold() if text else None

    def _matches_document(self, result: Dict[str, Any], document: Optional[DocumentRef]) -> bool:
        if document is None:
            return False

        requested_id = self._normalize_match_value(document.id)
        if requested_id and self._normalize_match_value(result.get(self._doc_id_field)) == requested_id:
            return True

        requested_name = self._normalize_match_value(document.name)
        if requested_name and self._normalize_match_value(result.get(self._doc_name_field)) == requested_name:
            return True

        return False

    def _result_key(self, result: Dict[str, Any]) -> str:
        chunk_key = result.get(self._chunk_key_field)
        if chunk_key:
            return f"chunk:{chunk_key}"

        excerpt = self._make_excerpt(result) or ""
        return "citation:{doc_id}|{doc_name}|{excerpt}".format(
            doc_id=result.get(self._doc_id_field) or "",
            doc_name=result.get(self._doc_name_field) or "",
            excerpt=excerpt,
        )

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

    async def _collect_results(self, pageable) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        async for item in pageable:
            result: Dict[str, Any] = dict(item)
            score = result.get("@search.score")
            if self._min_citation_score > 0 and (score is None or score < self._min_citation_score):
                logger.debug("Skipping citation with score=%s (below threshold %s)", score, self._min_citation_score)
                continue
            results.append(result)
        return results

    async def _search_results(
        self,
        *,
        client: SearchClient,
        query: str,
        top_k: int,
        filter_expr: Optional[str],
        allow_vector: bool,
    ) -> List[Dict[str, Any]]:
        query_type = "semantic" if self._semantic_config and query != "*" else "simple"
        semantic_configuration_name = self._semantic_config if query_type == "semantic" else None

        search_kwargs: Dict[str, Any] = {
            "search_text": query,
            "top": top_k,
            "query_type": query_type,
            "semantic_configuration_name": semantic_configuration_name,
            "select": list(
                {
                    self._chunk_key_field,
                    self._content_field,
                    self._doc_id_field,
                    self._doc_name_field,
                }
            ),
            "highlight_fields": self._content_field if query != "*" else None,
            "filter": filter_expr,
        }

        if allow_vector and self._vector_query_enabled and self._vector_field and query != "*":
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
                return await self._collect_results(vector_results)
            except Exception as e:
                if self._is_vector_profile_error(e):
                    self._vector_query_enabled = False
                    logger.warning(
                        "Vector query disabled for this runtime due to index vector profile mismatch; using plain search fallback."
                    )
                else:
                    logger.warning("Vector query failed; retrying without vector query: %s", e)

        plain_results = await client.search(**search_kwargs)
        return await self._collect_results(plain_results)

    def _to_citations(self, results: List[Dict[str, Any]]) -> List[Citation]:
        citations: List[Citation] = []
        for idx, result in enumerate(results, start=1):
            citations.append(
                Citation(
                    document_name=result.get(self._doc_name_field),
                    document_id=result.get(self._doc_id_field),
                    rank=idx,
                    score=result.get("@search.score"),
                    excerpt=self._make_excerpt(result),
                )
            )
        return citations

    async def _list_existing_chunk_keys(self, *, document_id: str) -> List[str]:
        client = self._get_client()
        filter_field = self._document_filter_field()
        if client is None or not filter_field:
            return []

        try:
            pageable = await client.search(
                search_text="*",
                top=1000,
                query_type="simple",
                select=[self._chunk_key_field],
                filter=f"{filter_field} eq '{_escape_odata_string(document_id)}'",
            )
        except Exception as exc:
            logger.warning("Unable to inspect existing chunks for document %s: %s", document_id, exc)
            return []

        keys: List[str] = []
        async for item in pageable:
            result = dict(item)
            key = result.get(self._chunk_key_field)
            if isinstance(key, str) and key:
                keys.append(key)
        return keys

    async def search(self, *, query: str, document: Optional[DocumentRef], top_k: int) -> List[Citation]:
        client = self._get_client()
        if client is None:
            logger.warning("Azure AI Search is not configured; returning empty citations")
            return []

        filter_expr = self._build_filter(document)
        try:
            preferred_results: List[Dict[str, Any]] = []
            if document is not None:
                if filter_expr:
                    preferred_results = await self._search_results(
                        client=client,
                        query=query,
                        top_k=1,
                        filter_expr=filter_expr,
                        allow_vector=True,
                    )
                    if not preferred_results:
                        preferred_results = await self._search_results(
                            client=client,
                            query="*",
                            top_k=1,
                            filter_expr=filter_expr,
                            allow_vector=False,
                        )

                global_results = await self._search_results(
                    client=client,
                    query=query,
                    top_k=top_k,
                    filter_expr=None,
                    allow_vector=True,
                )

                if not preferred_results:
                    for result in global_results:
                        if self._matches_document(result, document):
                            preferred_results = [result]
                            break

                merged: List[Dict[str, Any]] = []
                seen: Set[str] = set()
                for result in preferred_results + global_results:
                    key = self._result_key(result)
                    if key in seen:
                        continue
                    seen.add(key)
                    merged.append(result)
                    if len(merged) >= top_k:
                        break

                return self._to_citations(merged)

            plain_results = await self._search_results(
                client=client,
                query=query,
                top_k=top_k,
                filter_expr=None,
                allow_vector=True,
            )
            return self._to_citations(plain_results)
        except Exception as e:
            logger.error("Search failed after vector fallback: %s", e, exc_info=True)
            return []

    async def preview_document(self, *, document: DocumentRef, top_k: int) -> List[Citation]:
        client = self._get_client()
        if client is None:
            logger.warning("Azure AI Search is not configured; returning empty citations")
            return []

        filter_expr = self._build_filter(document)
        try:
            if filter_expr:
                results = await self._search_results(
                    client=client,
                    query="*",
                    top_k=top_k,
                    filter_expr=filter_expr,
                    allow_vector=False,
                )
                return self._to_citations(results)

            results = await self._search_results(
                client=client,
                query="*",
                top_k=max(top_k * 3, top_k),
                filter_expr=None,
                allow_vector=False,
            )
            filtered = [result for result in results if self._matches_document(result, document)]
            return self._to_citations(filtered[:top_k])
        except Exception as e:
            logger.error("Document preview failed: %s", e, exc_info=True)
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

        existing_keys = set(await self._list_existing_chunk_keys(document_id=document_id))

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

        new_keys = {
            str(doc[self._chunk_key_field])
            for doc in docs
            if doc.get(self._chunk_key_field) is not None
        }

        # Batch to avoid request limits.
        batch_size = 200
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            await client.merge_or_upload_documents(documents=batch)

        stale_keys = [key for key in existing_keys if key not in new_keys]
        for i in range(0, len(stale_keys), batch_size):
            batch = stale_keys[i : i + batch_size]
            await client.delete_documents(
                documents=[{self._chunk_key_field: key} for key in batch]
            )

    async def delete_document_chunks(self, *, document_id: str) -> None:
        client = self._get_client()
        if client is None:
            logger.warning("Azure AI Search is not configured; skipping document deletion")
            return

        existing_keys = await self._list_existing_chunk_keys(document_id=document_id)
        if not existing_keys:
            return

        batch_size = 200
        for i in range(0, len(existing_keys), batch_size):
            batch = existing_keys[i : i + batch_size]
            await client.delete_documents(
                documents=[{self._chunk_key_field: key} for key in batch]
            )
