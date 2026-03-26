from __future__ import annotations

import argparse
import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import get_settings
from models.infosearch_api import DocumentRef
from scripts.create_search_index import main as ensure_search_index
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.llm_service import AzureOpenAILLMService
from services.text_extraction import extract_text


@dataclass(frozen=True)
class ValidationDocument:
    document_id: str
    document_name: str
    content: str


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _build_documents(run_id: str) -> tuple[ValidationDocument, ValidationDocument]:
    target = ValidationDocument(
        document_id=f"validation/target-{run_id}.txt",
        document_name=f"target-{run_id}.txt",
        content=(
            "Validation target document for citation testing. "
            "Unique marker ORBIT-RED-77 appears only in this document. "
            "The retention window for this document is 17 days. "
            "This file is the preferred source when a direct document object is supplied."
        ),
    )
    other = ValidationDocument(
        document_id=f"validation/other-{run_id}.txt",
        document_name=f"other-{run_id}.txt",
        content=(
            "Validation comparison document for citation testing. "
            "Unique marker ORBIT-BLUE-22 appears only in this document. "
            "The retention window for this document is 33 days. "
            "This file exists to confirm that general search can still return other relevant documents."
        ),
    )
    return target, other


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _is_supported_blob(name: str) -> bool:
    lower = (name or "").lower()
    return lower.endswith(".pdf") or lower.endswith(".txt") or lower.endswith(".md") or lower.endswith(".csv")


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _candidate_queries(text: str) -> list[str]:
    cleaned = _normalize_whitespace(text)
    if not cleaned:
        return []

    queries: list[str] = []

    exact_phrases = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-/,()]{3,}(?:\s+[A-Za-z0-9][A-Za-z0-9\-/,()]{3,}){3,11}", cleaned)
    for phrase in exact_phrases[:20]:
        normalized = phrase.strip(" .,;:-")
        if len(normalized.split()) >= 4 and normalized not in queries:
            queries.append(normalized)

    uppercase_tokens = [token.strip(".,;:()[]{}") for token in cleaned.split() if any(ch.isalpha() for ch in token)]
    strong_terms = [token for token in uppercase_tokens if sum(ch.isupper() for ch in token) >= 3 or any(ch.isdigit() for ch in token)]
    for token in strong_terms[:10]:
        if token not in queries:
            queries.append(token)

    if len(cleaned.split()) >= 8:
        fallback = " ".join(cleaned.split()[:8])
        if fallback not in queries:
            queries.append(fallback)

    return queries[:25]


def _print_citations(citations) -> None:
    if not citations:
        print("No citations returned.")
        return

    for citation in citations:
        print(
            "- rank={rank} doc_id={doc_id} doc_name={doc_name} score={score} excerpt={excerpt}".format(
                rank=citation.rank,
                doc_id=citation.document_id,
                doc_name=citation.document_name,
                score=(f"{citation.score:.4f}" if isinstance(citation.score, (int, float)) else citation.score),
                excerpt=(citation.excerpt or "").replace("\n", " "),
            )
        )


async def _delete_documents(search_service: AzureAISearchService, document_ids: Iterable[str]) -> None:
    client = search_service._get_client()
    if client is None:
        return

    key_field = search_service._chunk_key_field
    chunk_ids = [{key_field: f"chunk_{hashlib.sha256(document_id.encode('utf-8')).hexdigest()}_0"} for document_id in document_ids]
    if chunk_ids:
        await client.delete_documents(documents=chunk_ids)


async def _wait_until_searchable(
    search_service: AzureAISearchService,
    *,
    document: ValidationDocument,
    query: str,
    timeout_seconds: int,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    ref = DocumentRef(id=document.document_id, name=document.document_name)

    while True:
        citations = await search_service.search(query=query, document=ref, top_k=3)
        if any(citation.document_id == document.document_id for citation in citations):
            return
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError(f"Timed out waiting for document {document.document_id} to become searchable")
        await asyncio.sleep(2)


async def _index_documents(search_service: AzureAISearchService, docs: Iterable[ValidationDocument]) -> None:
    for doc in docs:
        await search_service.upsert_chunks(
            document_id=doc.document_id,
            document_name=doc.document_name,
            chunks=[
                {
                    "id": f"chunk_{hashlib.sha256(doc.document_id.encode('utf-8')).hexdigest()}_0",
                    "content": doc.content,
                    "chunkIndex": 0,
                    "chunkStart": 0,
                    "chunkEnd": len(doc.content),
                }
            ],
        )


async def _find_live_blob_document(
    *,
    settings,
    search_service: AzureAISearchService,
    blob_limit: int,
    query_limit: int,
) -> tuple[ValidationDocument, str, list] | None:
    if not settings.azure_storage_connection_string or not settings.azure_storage_container:
        raise RuntimeError("Azure Blob Storage is not configured. Set AZURE_STORAGE_CONNECTION_STRING and AZURE_STORAGE_CONTAINER.")

    blob_source = BlobStorageSource(
        connection_string=settings.azure_storage_connection_string,
        container=settings.azure_storage_container,
        prefix=settings.azure_storage_prefix,
    )
    await blob_source.open()
    try:
        seen = 0
        async for blob in blob_source.list_blobs(limit=blob_limit):
            if not _is_supported_blob(blob.name):
                continue

            seen += 1
            content = await blob_source.download(blob_name=blob.name)
            text = extract_text(blob_name=blob.name, content=content)
            normalized_text = _normalize_whitespace(text or "")
            if len(normalized_text) < 80:
                continue

            document = ValidationDocument(
                document_id=blob.name,
                document_name=blob.name.split("/")[-1],
                content=normalized_text,
            )
            document_ref = DocumentRef(id=document.document_id, name=document.document_name)

            for query in _candidate_queries(normalized_text)[:query_limit]:
                doc_scoped = await search_service.search(query=query, document=document_ref, top_k=5)
                if not any(c.document_id == document.document_id for c in doc_scoped):
                    continue

                unscoped = await search_service.search(query=query, document=None, top_k=5)
                if any(c.document_id == document.document_id for c in unscoped):
                    return document, query, unscoped

        return None
    finally:
        await blob_source.close()


async def _run_validation(args) -> int:
    settings = get_settings()
    search_service = AzureAISearchService.from_settings(settings)
    llm_service = AzureOpenAILLMService.from_settings(settings)

    if not search_service._filter_doc_id_field:
        search_service._filter_doc_id_field = search_service._doc_id_field
    if not search_service._filter_doc_name_field:
        search_service._filter_doc_name_field = search_service._doc_name_field

    if args.ensure_index:
        ensure_search_index()

    await llm_service.open()
    cleanup_ids: list[str] = []

    try:
        if args.source == "blob":
            _print_header("Finding a real blob document that is searchable in Azure AI Search")
            live_match = await _find_live_blob_document(
                settings=settings,
                search_service=search_service,
                blob_limit=args.blob_limit,
                query_limit=args.query_limit,
            )
            if live_match is None:
                print("FAIL: could not find a real supported blob whose extracted content is searchable both with and without a document filter")
                return 1

            document, query, initial_citations = live_match
            print(f"Selected blob: {document.document_id}")
            print(f"Derived query: {query}")

            no_document_question = f"Answer using the cited sources: {query}"
            with_document_question = f"Answer using this document only: {query}"

            _print_header("Scenario 1: query actual AI Search without document object")
            citations = initial_citations
            _print_citations(citations)
            if llm_service._client is not None:
                answer = await llm_service.answer(question=no_document_question, citations=citations, history=None)
                print("Answer:")
                print(answer)
            else:
                print("Azure OpenAI is not configured; skipped answer generation.")

            document_ref = DocumentRef(id=document.document_id, name=document.document_name)
            _print_header("Scenario 2: query actual AI Search with document object")
            citations_with_doc = await search_service.search(query=query, document=document_ref, top_k=args.top_k)
            _print_citations(citations_with_doc)
            if llm_service._client is not None:
                answer_with_doc = await llm_service.answer(
                    question=with_document_question,
                    citations=citations_with_doc,
                    history=None,
                )
                print("Answer:")
                print(answer_with_doc)
            else:
                print("Azure OpenAI is not configured; skipped answer generation.")

            _print_header("Validation checks")
            if not citations:
                print("FAIL: no citations returned for the real-data unscoped query")
                return 1
            if not any(c.document_id == document.document_id for c in citations):
                print("FAIL: real blob document was not returned in the unscoped citations")
                return 1
            print("PASS: unscoped real-data query returned the real blob document in citations")

            if not citations_with_doc:
                print("FAIL: no citations returned for the real-data document-scoped query")
                return 1
            if citations_with_doc[0].document_id != document.document_id:
                print(
                    "FAIL: real-data document-scoped query did not rank the referenced document first; got {doc_id}".format(
                        doc_id=citations_with_doc[0].document_id
                    )
                )
                return 1
            print("PASS: document-scoped real-data query ranked the referenced blob document as citation 1")
            return 0

        run_id = _now_stamp()
        target_doc, other_doc = _build_documents(run_id)
        cleanup_ids.extend([target_doc.document_id, other_doc.document_id])

        _print_header("Indexing temporary validation documents")
        await _index_documents(search_service, [target_doc, other_doc])
        print(f"Indexed {target_doc.document_id}")
        print(f"Indexed {other_doc.document_id}")

        await _wait_until_searchable(
            search_service,
            document=target_doc,
            query="ORBIT-RED-77",
            timeout_seconds=args.timeout_seconds,
        )
        await _wait_until_searchable(
            search_service,
            document=other_doc,
            query="ORBIT-BLUE-22",
            timeout_seconds=args.timeout_seconds,
        )

        no_document_search_query = "retention window"
        no_document_question = "Which document mentions ORBIT-RED-77 and what retention window is stated?"
        with_document_search_query = "retention window"
        with_document_question = "What retention window is stated in this document?"

        _print_header("Scenario 1: query without document object")
        citations = await search_service.search(query=no_document_search_query, document=None, top_k=args.top_k)
        _print_citations(citations)
        if llm_service._client is not None:
            answer = await llm_service.answer(question=no_document_question, citations=citations, history=None)
            print("Answer:")
            print(answer)
        else:
            print("Azure OpenAI is not configured; skipped answer generation.")

        target_ref = DocumentRef(id=target_doc.document_id, name=target_doc.document_name)
        _print_header("Scenario 2: query with document object")
        citations_with_doc = await search_service.search(
            query=with_document_search_query,
            document=target_ref,
            top_k=args.top_k,
        )
        _print_citations(citations_with_doc)
        if llm_service._client is not None:
            answer_with_doc = await llm_service.answer(
                question=with_document_question,
                citations=citations_with_doc,
                history=None,
            )
            print("Answer:")
            print(answer_with_doc)
        else:
            print("Azure OpenAI is not configured; skipped answer generation.")

        _print_header("Validation checks")
        if not citations:
            print("FAIL: no citations returned for the no-document query")
            return 1
        if not any(c.document_id == target_doc.document_id for c in citations):
            print("FAIL: target document was not returned in the no-document query")
            return 1
        print("PASS: no-document query returned the target document in the citations")

        if not citations_with_doc:
            print("FAIL: no citations returned for the document-scoped query")
            return 1
        if citations_with_doc[0].document_id != target_doc.document_id:
            print(
                "FAIL: direct document query did not rank the referenced document first; got {doc_id}".format(
                    doc_id=citations_with_doc[0].document_id
                )
            )
            return 1
        print("PASS: direct document query ranked the referenced document as citation 1")
        return 0
    finally:
        if cleanup_ids and not args.keep_docs:
            _print_header("Cleaning up temporary validation documents")
            await _delete_documents(search_service, cleanup_ids)
            for document_id in cleanup_ids:
                print(f"Deleted indexed chunk for {document_id}")
        await llm_service.close()
        await search_service.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate citation ordering with and without a direct document reference.")
    parser.add_argument(
        "--source",
        choices=["blob", "synthetic"],
        default="blob",
        help="Use real blob data or temporary synthetic validation documents",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of citations to request from search")
    parser.add_argument("--blob-limit", type=int, default=20, help="Maximum number of blobs to inspect when using --source blob")
    parser.add_argument("--query-limit", type=int, default=12, help="Maximum derived queries to test per blob when using --source blob")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=45,
        help="How long to wait for uploaded validation documents to become searchable",
    )
    parser.add_argument(
        "--ensure-index",
        action="store_true",
        help="Create or update the Azure AI Search index before running validation",
    )
    parser.add_argument(
        "--keep-docs",
        action="store_true",
        help="Keep temporary validation documents in the index after the run",
    )
    args = parser.parse_args()

    load_dotenv()
    return asyncio.run(_run_validation(args))


if __name__ == "__main__":
    raise SystemExit(main())