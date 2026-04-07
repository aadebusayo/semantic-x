import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.dependencies import get_blob_source, get_chat_repo, get_document_catalog, get_llm_service, get_search_service, get_signalr_service, get_suggestions_repo, get_tts_service
from config import get_settings
from models.infosearch_api import (
	ChatHistoryItem,
	ChatMessageRequest,
	ChatMessageResponse,
	ChatTranscript,
	ChatTurn,
	DocumentRef,
	DocumentPreviewRequest,
	DocumentPreviewResponse,
	IndexedDocument,
	QuickQuestionsCreateRequest,
	QuickQuestionsResponse,
	ReactionRequest,
	RecentQuickQuestionsItem,
	RenameChatRequest,
)
from services.azure_ai_search import AzureAISearchService
from services.blob_storage import BlobStorageSource
from services.chat_engine import make_title
from services.cosmos_repositories import CosmosChatRepository, CosmosSuggestionsRepository
from services.document_catalog import DocumentCatalogService
from services.llm_service import AzureOpenAILLMService
from services.signalr_service import SignalRService
from services.suggestions_engine import build_suggestion_title, generate_quick_questions
from services.tts_service import AzureTTSService


router = APIRouter(tags=["infosearch"])


def _sse(event: str, data: dict) -> str:
	return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@router.post("/chat/messages", response_model=ChatMessageResponse)
async def post_chat_message(
	payload: ChatMessageRequest,
	search_service: AzureAISearchService = Depends(get_search_service),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
	llm_service: AzureOpenAILLMService = Depends(get_llm_service),
):
	message = payload.message.strip()
	if not message:
		raise HTTPException(status_code=400, detail="Use /api/v1/documents/preview for document-only requests")

	settings = get_settings()
	chat_id = payload.chat_id or str(uuid.uuid4())
	user_id = payload.user_id or settings.default_user_id
	top_k = payload.top_k or settings.default_top_k
	document = await document_catalog.normalize_document_ref(payload.document) if payload.document else None

	citations, existing = await asyncio.gather(
		search_service.search(query=message, document=document, top_k=top_k),
		chat_repo.get_chat(chat_id=chat_id, user_id=user_id),
	)
	citations = await document_catalog.normalize_citations(citations=citations)

	# Use existing title only if it is a real AI-generated one (not a placeholder / fallback).
	existing_title = (existing or {}).get("title")
	_is_placeholder_title = not existing_title or existing_title.strip().lower() in {"new chat", ""}

	# Build conversation history from the last 5 exchanges (10 turns) for LLM context.
	prior_turns = (existing or {}).get("turns") or []
	history = [
		{"role": t["role"], "content": t["content"]}
		for t in prior_turns[-10:]
		if t.get("role") in {"user", "assistant"} and t.get("content")
	]

	try:
		# Run title generation and answer in parallel to minimise latency.
		if _is_placeholder_title:
			title_result, answer = await asyncio.gather(
				llm_service.generate_title(user_message=message),
				llm_service.answer(question=message, citations=citations, history=history),
				return_exceptions=True,
			)
			if isinstance(title_result, Exception):
				logger.warning("Title generation failed, using fallback: %s", title_result)
				title = make_title(message)
			else:
				title = title_result
			if isinstance(answer, Exception):
				raise answer
		else:
			title = existing_title
			answer = await llm_service.answer(question=message, citations=citations, history=history)
	except RuntimeError as exc:
		raise HTTPException(status_code=503, detail=str(exc)) from exc

	stored = await chat_repo.append_turns(
		chat_id=chat_id,
		user_id=user_id,
		title=title,
		user_message=message,
		assistant_message=answer,
	)

	created_at = datetime.fromisoformat(stored["updatedAt"])
	return ChatMessageResponse(
		chat_id=chat_id,
		title=stored.get("title") or title,
		answer=answer,
		citations=citations,
		created_at=created_at,
	)


@router.post("/chat/messages/stream")
async def post_chat_message_stream(
	payload: ChatMessageRequest,
	search_service: AzureAISearchService = Depends(get_search_service),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
	llm_service: AzureOpenAILLMService = Depends(get_llm_service),
	signalr_service: SignalRService = Depends(get_signalr_service),
):
	message = payload.message.strip()
	if not message:
		raise HTTPException(status_code=400, detail="Use /api/v1/documents/preview for document-only requests")

	settings = get_settings()
	chat_id = payload.chat_id or str(uuid.uuid4())
	user_id = payload.user_id or settings.default_user_id
	top_k = payload.top_k or settings.default_top_k
	document = await document_catalog.normalize_document_ref(payload.document) if payload.document else None

	citations, existing = await asyncio.gather(
		search_service.search(query=message, document=document, top_k=top_k),
		chat_repo.get_chat(chat_id=chat_id, user_id=user_id),
	)
	citations = await document_catalog.normalize_citations(citations=citations)
	existing_title = (existing or {}).get("title")
	_is_placeholder_title = not existing_title or existing_title.strip().lower() in {"new chat", ""}
	title = None if _is_placeholder_title else existing_title

	prior_turns = (existing or {}).get("turns") or []
	history = [
		{"role": t["role"], "content": t["content"]}
		for t in prior_turns[-10:]
		if t.get("role") in {"user", "assistant"} and t.get("content")
	]

	async def event_generator():
		nonlocal title
		answer_parts: list[str] = []

		if not title:
			try:
				title = await llm_service.generate_title(user_message=message)
			except Exception:
				title = make_title(message)

		meta_payload = {
			"chat_id": chat_id,
			"title": title,
			"citations": [c.model_dump() for c in citations],
		}
		yield _sse("meta", meta_payload)
		await signalr_service.publish(user_id=user_id, event="meta", payload=meta_payload)

		try:
			async for token in llm_service.stream_answer(question=message, citations=citations, history=history):
				answer_parts.append(token)
				chunk_payload = {"chat_id": chat_id, "delta": token}
				yield _sse("chunk", chunk_payload)
				await signalr_service.publish(user_id=user_id, event="chunk", payload=chunk_payload)

			answer = "".join(answer_parts).strip() or "I could not generate a response."
			stored = await chat_repo.append_turns(
				chat_id=chat_id,
				user_id=user_id,
				title=title or make_title(message),
				user_message=message,
				assistant_message=answer,
			)

			complete_payload = {
				"chat_id": chat_id,
				"title": stored.get("title") or title,
				"created_at": datetime.fromisoformat(stored["updatedAt"]).isoformat(),
			}
			yield _sse("complete", complete_payload)
			await signalr_service.publish(user_id=user_id, event="complete", payload=complete_payload)
		except RuntimeError as exc:
			error_payload = {"chat_id": chat_id, "error": str(exc)}
			yield _sse("error", error_payload)
			await signalr_service.publish(user_id=user_id, event="error", payload=error_payload)
		except Exception as exc:
			error_payload = {"chat_id": chat_id, "error": f"stream_failed: {exc}"}
			yield _sse("error", error_payload)
			await signalr_service.publish(user_id=user_id, event="error", payload=error_payload)

	return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/documents", response_model=list[IndexedDocument])
async def list_documents(
	limit: int = Query(default=100, ge=1, le=500),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
):
	return await document_catalog.list_documents(limit=limit)


@router.post("/documents/preview", response_model=DocumentPreviewResponse)
async def preview_document(
	payload: DocumentPreviewRequest,
	search_service: AzureAISearchService = Depends(get_search_service),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
):
	document = await document_catalog.normalize_document_ref(payload.document)
	if not document.id and not document.name:
		raise HTTPException(status_code=400, detail="Document id or name is required")

	settings = get_settings()
	top_k = payload.top_k or settings.default_top_k
	citations = await search_service.preview_document(document=document, top_k=top_k)
	citations = await document_catalog.normalize_citations(citations=citations)
	return DocumentPreviewResponse(
		document=document,
		citations=citations,
		created_at=datetime.now(timezone.utc),
	)


@router.get("/chats", response_model=list[ChatHistoryItem])
async def list_chats(
	user_id: Optional[str] = Query(default=None),
	limit: int = Query(default=20, ge=1, le=100),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
):
	items = await chat_repo.list_chats(user_id=user_id, limit=limit)
	out: list[ChatHistoryItem] = []
	for it in items:
		updated = it.get("updatedAt") or it.get("updated_at")
		if not updated:
			continue
		out.append(
			ChatHistoryItem(
				chat_id=it.get("chatId") or it.get("id"),
				title=it.get("title") or "New chat",
				updated_at=datetime.fromisoformat(updated),
			)
		)
	return out


@router.get("/chats/{chat_id}", response_model=ChatTranscript)
async def get_chat(
	chat_id: str,
	user_id: Optional[str] = Query(default=None),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
):
	settings = get_settings()
	uid = user_id or settings.default_user_id
	item = await chat_repo.get_chat(chat_id=chat_id, user_id=uid)
	if item is None:
		raise HTTPException(status_code=404, detail="Chat not found")

	turns: list[ChatTurn] = []
	for t in item.get("turns") or []:
		try:
			turns.append(
				ChatTurn(
					role=t.get("role"),
					content=t.get("content") or "",
					created_at=datetime.fromisoformat(t.get("createdAt")),
				)
			)
		except Exception:
			continue

	return ChatTranscript(
		chat_id=item.get("chatId") or item.get("id"),
		title=item.get("title") or "New chat",
		created_at=datetime.fromisoformat(item.get("createdAt")),
		updated_at=datetime.fromisoformat(item.get("updatedAt")),
		turns=turns,
	)


@router.post("/documents/suggestions", response_model=QuickQuestionsResponse)
async def create_document_suggestions(
	payload: QuickQuestionsCreateRequest,
	suggestions_repo: CosmosSuggestionsRepository = Depends(get_suggestions_repo),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
	llm_service: AzureOpenAILLMService = Depends(get_llm_service),
):
	document = await document_catalog.normalize_document_ref(payload.document)
	blob_name = await document_catalog.resolve_blob_name(document)
	document_label = document.name or document.id
	title_fallback = build_suggestion_title(document_name=document_label, text=payload.text)
	questions_task = llm_service.generate_suggestions(
		document_name=document_label,
		text=payload.text,
		limit=3,
	)
	title_generator = getattr(llm_service, "generate_suggestion_title", None)
	title_task = (
		title_generator(document_name=document_label, text=payload.text)
		if callable(title_generator)
		else asyncio.sleep(0, result=title_fallback)
	)
	questions, title = await asyncio.gather(questions_task, title_task)

	# Try LLM-based suggestions first
	title = (title or title_fallback).strip() or title_fallback

	# Fallback to heuristic if LLM fails or returns empty
	if not questions:
		questions = generate_quick_questions(
			document_name=document_label,
			text=payload.text,
			limit=3,
		)

	try:
		item = await suggestions_repo.upsert_questions(
			document_id=document.id,
			document_name=document.name,
			title=title,
			blob_name=blob_name,
			questions=questions,
		)
	except TypeError:
		item = await suggestions_repo.upsert_questions(
			document_id=document.id,
			document_name=document.name,
			blob_name=blob_name,
			questions=questions,
		)
		item["title"] = title
	return QuickQuestionsResponse(
		document=document,
		title=item.get("title") or title,
		questions=item.get("questions") or questions,
		created_at=datetime.fromisoformat(item["createdAt"]),
	)


@router.get("/suggestions/recent", response_model=list[RecentQuickQuestionsItem])
async def list_recent_suggestions(
	limit: int = Query(default=3, ge=1, le=10),
	suggestions_repo: CosmosSuggestionsRepository = Depends(get_suggestions_repo),
	document_catalog: DocumentCatalogService = Depends(get_document_catalog),
	blob_source: BlobStorageSource | None = Depends(get_blob_source),
):
	items = await suggestions_repo.list_recent(limit=min(limit * 5, 50))
	out: list[RecentQuickQuestionsItem] = []
	for it in items:
		blob_name = it.get("blobName")
		if blob_source is not None and blob_name:
			try:
				if not await blob_source.exists(blob_name=blob_name):
					await suggestions_repo.delete_questions(
						document_id=it.get("documentId"),
						document_name=it.get("documentName"),
					)
					continue
			except Exception:
				pass

		document = await document_catalog.normalize_document_ref(
			DocumentRef(id=it.get("documentId"), name=it.get("documentName"))
		)
		title = (it.get("title") or "").strip() or build_suggestion_title(
			document_name=document.name or it.get("documentName") or it.get("documentId"),
		)
		out.append(
			RecentQuickQuestionsItem(
				document=document,
				title=title,
				questions=it.get("questions") or [],
				created_at=datetime.fromisoformat(it.get("createdAt")),
			)
		)
		if len(out) >= limit:
			break
	return out


# ── Rename chat ──────────────────────────────────────────────────────────────

@router.patch("/chats/{chat_id}/title")
async def rename_chat(
	chat_id: str,
	payload: RenameChatRequest,
	user_id: Optional[str] = Query(default=None),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
):
	settings = get_settings()
	uid = user_id or settings.default_user_id
	item = await chat_repo.rename_chat(chat_id=chat_id, user_id=uid, new_title=payload.title)
	if item is None:
		raise HTTPException(status_code=404, detail="Chat not found")
	return {"chat_id": chat_id, "title": item["title"]}


# ── Turn reaction ─────────────────────────────────────────────────────────────

@router.post("/chats/{chat_id}/turns/{turn_index}/reaction")
async def save_turn_reaction(
	chat_id: str,
	turn_index: int,
	payload: ReactionRequest,
	user_id: Optional[str] = Query(default=None),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
):
	settings = get_settings()
	uid = user_id or settings.default_user_id
	item = await chat_repo.save_reaction(
		chat_id=chat_id,
		user_id=uid,
		turn_index=turn_index,
		reaction=payload.reaction,
	)
	if item is None:
		raise HTTPException(status_code=404, detail="Chat or turn not found")
	return {"chat_id": chat_id, "turn_index": turn_index, "reaction": payload.reaction}


# ── Turn TTS audio ────────────────────────────────────────────────────────────

@router.get("/chats/{chat_id}/turns/{turn_index}/audio")
async def get_turn_audio(
	chat_id: str,
	turn_index: int,
	user_id: Optional[str] = Query(default=None),
	chat_repo: CosmosChatRepository = Depends(get_chat_repo),
	tts_service: AzureTTSService = Depends(get_tts_service),
):
	from fastapi.responses import Response as FResponse

	if not tts_service.enabled:
		raise HTTPException(status_code=503, detail="TTS service not configured")

	settings = get_settings()
	uid = user_id or settings.default_user_id
	item = await chat_repo.get_chat(chat_id=chat_id, user_id=uid)
	if item is None:
		raise HTTPException(status_code=404, detail="Chat not found")

	turns = item.get("turns") or []
	if turn_index < 0 or turn_index >= len(turns):
		raise HTTPException(status_code=404, detail="Turn not found")

	turn = turns[turn_index]
	if turn.get("role") != "assistant":
		raise HTTPException(status_code=400, detail="Turn is not an assistant message")

	audio = await tts_service.synthesize_for_turn(
		text=turn.get("content") or "",
		chat_id=chat_id,
		turn_index=turn_index,
	)
	if audio is None:
		raise HTTPException(status_code=502, detail="TTS synthesis failed")

	return FResponse(content=audio, media_type="audio/mpeg")
