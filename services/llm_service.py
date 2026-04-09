from __future__ import annotations

import inspect
import logging
import re
from typing import AsyncIterator, List, Optional

from openai import AsyncAzureOpenAI

from config import Settings
from models.infosearch_api import Citation
from services.suggestions_engine import build_suggestion_title

logger = logging.getLogger(__name__)


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    text = value.strip()
    return (not text) or ("<" in text and ">" in text)


def _is_unsupported_temperature_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "temperature" in message and "unsupported value" in message


def _is_unsupported_reasoning_effort_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "reasoning_effort" in message and ("unsupported" in message or "unknown" in message)


def _is_hidden_reasoning_exhaustion(response) -> bool:
    if not response.choices:
        return False
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    content = (getattr(message, "content", None) or "").strip() if message is not None else ""
    if content:
        return False
    if getattr(choice, "finish_reason", None) != "length":
        return False

    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    return completion_tokens > 0 and reasoning_tokens >= completion_tokens


class AzureOpenAILLMService:
    def __init__(
        self,
        *,
        endpoint: Optional[str],
        api_key: Optional[str],
        api_version: str,
        deployment: Optional[str],
        temperature: float,
        max_tokens: int,
    ):
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._deployment = deployment
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._client: Optional[AsyncAzureOpenAI] = None

    def _require_configured(self) -> None:
        if self._client is None or not self._deployment:
            raise RuntimeError(
                "Azure OpenAI is required for chat. Set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_CHAT_DEPLOYMENT."
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> "AzureOpenAILLMService":
        return cls(
            endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            deployment=settings.azure_openai_chat_deployment,
            temperature=settings.azure_openai_temperature,
            max_tokens=settings.azure_openai_max_tokens,
        )

    def _configured(self) -> bool:
        return not (
            _is_placeholder(self._endpoint)
            or _is_placeholder(self._api_key)
            or _is_placeholder(self._deployment)
        )

    async def open(self) -> None:
        if not self._configured():
            self._client = None
            return
        self._client = AsyncAzureOpenAI(
            azure_endpoint=self._endpoint,
            api_key=self._api_key,
            api_version=self._api_version,
        )

    async def close(self) -> None:
        if self._client is None:
            return
        close_fn = getattr(self._client, "close", None)
        if close_fn is not None:
            maybe_result = close_fn()
            if inspect.isawaitable(maybe_result):
                await maybe_result
        self._client = None

    async def _create_chat_completion(self, **kwargs):
        assert self._client is not None
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            if "temperature" not in kwargs or not _is_unsupported_temperature_error(exc):
                if "reasoning_effort" not in kwargs or not _is_unsupported_reasoning_effort_error(exc):
                    raise
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("reasoning_effort", None)
                logger.info("Retrying Azure OpenAI completion without reasoning_effort for deployment=%r", self._deployment)
                return await self._client.chat.completions.create(**retry_kwargs)
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("temperature", None)
            logger.info("Retrying Azure OpenAI completion without temperature for deployment=%r", self._deployment)
            return await self._client.chat.completions.create(**retry_kwargs)

    @staticmethod
    def _sanitize_citations(text: str, citation_count: int) -> str:
        """Strip [N] markers where N is out of range to remove hallucinated refs."""
        if not citation_count:
            return re.sub(r'\[\d+\]', '', text).strip()
        def _replace(m: re.Match) -> str:
            n = int(m.group(1))
            return m.group(0) if 1 <= n <= citation_count else ''
        return re.sub(r'\[(\d+)\]', _replace, text).strip()

    @staticmethod
    def _citations_context(citations: List[Citation]) -> str:
        if not citations:
            return "No citations were retrieved from Azure AI Search."

        lines: List[str] = []
        for idx, citation in enumerate(citations[:6], start=1):
            lines.append(
                f"[{idx}] docId={citation.document_id or ''} docName={citation.document_name or ''} "
                f"score={citation.score if citation.score is not None else ''} excerpt={citation.excerpt or ''}"
            )
        return "\n".join(lines)

    async def answer(
        self,
        *,
        question: str,
        citations: List[Citation],
        history: Optional[List[dict]] = None,
    ) -> str:
        self._require_configured()

        system_prompt = (
            "You are Infosearch assistant — a helpful, friendly AI. "
            "Handle every type of message naturally:\n"
            "- Casual conversation (greetings, small talk): respond warmly and naturally.\n"
            "- General knowledge (math, facts, definitions): answer directly and concisely.\n"
            "- Document questions: use the provided document excerpts when relevant. "
            "Cite each source inline with its number, e.g. [1] or [2], placed directly after the sentence that uses it. "
            "Only cite numbers that exist in the provided list. "
            "If citations are present, prefer answering from them instead of falling back to generic knowledge. "
            "If the excerpts are only partially relevant, say that clearly and answer from the retrieved evidence first.\n"
            "- Questions needing private document context not in citations: say you couldn't find enough relevant information in the retrieved documents.\n"
            "- When using document excerpts, give a substantive answer: usually a short paragraph plus brief bullets only if they add clarity, not one-line fragments."
        )

        if citations:
            citations_block = self._citations_context(citations)
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Document excerpts (cite inline with [N] when used — only use numbers from this list):\n{citations_block}\n\n"
                "Write a grounded answer that synthesizes the excerpts into a useful response. "
                "Prefer 2-6 complete sentences or a short paragraph with bullets only when appropriate. "
                "Place [N] markers inline after each sentence that draws from a source. "
                "Do not answer with vague filler if the excerpts contain specifics."
            )
        else:
            user_prompt = f"Message:\n{question}"

        messages = [{"role": "system", "content": system_prompt}]
        # Inject prior conversation turns (last N, already trimmed by caller)
        for turn in (history or []):
            role = turn.get("role")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        logger.debug(
            "answer: sending %d messages (history=%d turns) to deployment=%r",
            len(messages), len(history or []), self._deployment,
        )

        response = await self._create_chat_completion(
            model=self._deployment,
            messages=messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
        )

        if _is_hidden_reasoning_exhaustion(response):
            logger.info(
                "answer: retrying with higher token budget and minimal reasoning for deployment=%r",
                self._deployment,
            )
            response = await self._create_chat_completion(
                model=self._deployment,
                messages=messages,
                temperature=self._temperature,
                max_completion_tokens=min(max(self._max_tokens * 2, 4000), 12000),
                reasoning_effort="minimal",
            )

        text = ""
        if response.choices and response.choices[0].message:
            msg = response.choices[0].message
            text = (msg.content or "").strip()

            if not text:
                finish_reason = response.choices[0].finish_reason
                refusal = getattr(msg, "refusal", None)
                logger.warning(
                    "answer: empty content — finish_reason=%r refusal=%r "
                    "usage=%r question=%r",
                    finish_reason,
                    refusal,
                    response.usage,
                    question[:120],
                )
                # Some GPT-5 preview builds surface the text in refusal when
                # the request is refused; surface that so the user sees why.
                if refusal:
                    return f"[Refused] {refusal}"

        if not text:
            logger.warning("answer: no choices or empty message for question=%r", question[:80])
            return "I could not generate a response. Please try again."

        return self._sanitize_citations(text, len(citations))

    async def generate_title(self, *, user_message: str) -> str:
        self._require_configured()

        prompt = (
            f"Write a short title (max 6 words) for this chat message. "
            f"Reply with the title only — no quotes, no punctuation at the end.\n\n"
            f"Message: {user_message[:300]}"
        )

        response = await self._create_chat_completion(
            model=self._deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=self._temperature,
            max_completion_tokens=1000,
        )

        text = ""
        if response.choices and response.choices[0].message:
            text = (response.choices[0].message.content or "").strip()

        if not text:
            logger.warning("generate_title: LLM returned empty content; falling back to message-based title")
            # Fall back to first 6 words of the question rather than generic "New chat"
            words = [w for w in user_message.strip().split() if w]
            return " ".join(words[:6])[:80] or "New chat"

        words = [w for w in text.split() if w]
        return " ".join(words[:6])[:80] or " ".join(user_message.strip().split()[:6])[:80] or "New chat"

    async def generate_suggestions(
        self,
        *,
        document_name: Optional[str],
        text: Optional[str],
        limit: int = 3,
    ) -> List[str]:
        """Generate short, contextual question suggestions based on document content."""
        if not self._configured() or self._client is None:
            return []

        name = (document_name or "this document").strip() or "this document"
        snippet = (text or "")[:1500]  # Limit context size

        if not snippet.strip():
            return []

        prompt = (
            f"Based on this document excerpt, generate exactly {limit} distinct user questions. "
            f"Each question must be specific to the document content, not a generic template. "
            f"Avoid repeating patterns like summary, key points, or what is X about unless the content truly demands it.\n\n"
            f"Document: {name}\n"
            f"Content:\n{snippet}\n\n"
            f"Return ONLY the {limit} questions, one per line, no numbering, no quotes, no extra text."
        )

        try:
            response = await self._create_chat_completion(
                model=self._deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=200,
            )

            if _is_hidden_reasoning_exhaustion(response):
                logger.info(
                    "generate_suggestions: retrying with higher token budget and minimal reasoning for deployment=%r",
                    self._deployment,
                )
                response = await self._create_chat_completion(
                    model=self._deployment,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_completion_tokens=600,
                    reasoning_effort="minimal",
                )

            text_response = ""
            if response.choices and response.choices[0].message:
                text_response = (response.choices[0].message.content or "").strip()

            if not text_response:
                retry_prompt = (
                    f"Read this document excerpt and write exactly {limit} concrete questions a user would ask after reading it. "
                    f"Make the questions varied and content-specific.\n\n"
                    f"Document: {name}\n"
                    f"Excerpt:\n{snippet}\n\n"
                    "Return only the questions, one per line."
                )
                response = await self._create_chat_completion(
                    model=self._deployment,
                    messages=[{"role": "user", "content": retry_prompt}],
                    temperature=0.7,
                    max_completion_tokens=220,
                )
                if _is_hidden_reasoning_exhaustion(response):
                    response = await self._create_chat_completion(
                        model=self._deployment,
                        messages=[{"role": "user", "content": retry_prompt}],
                        temperature=0.7,
                        max_completion_tokens=600,
                        reasoning_effort="minimal",
                    )
                if response.choices and response.choices[0].message:
                    text_response = (response.choices[0].message.content or "").strip()
                if not text_response:
                    return []

            # Parse lines and clean up
            lines = [line.strip() for line in text_response.split("\n") if line.strip()]
            # Remove any numbering prefixes like "1.", "1)", "-", etc.
            cleaned = []
            for line in lines[:limit]:
                # Strip common prefixes
                clean = re.sub(r'^[\d]+[.):\-]\s*', '', line)
                clean = re.sub(r'^[-*]\s*', '', clean)
                clean = clean.strip('"\'')
                if clean:
                    cleaned.append(clean)

            return cleaned[:limit]
        except Exception as e:
            logger.warning("generate_suggestions failed: %s", e)
            return []

    async def generate_suggestion_title(
        self,
        *,
        document_name: Optional[str],
        text: Optional[str],
    ) -> str:
        fallback = build_suggestion_title(document_name=document_name, text=text)
        if not self._configured() or self._client is None:
            return fallback

        prompt = (
            "Write a short, human-friendly document title in at most 5 words. "
            "Use the content to infer the title. Avoid file extensions, ids, GUIDs, and verbs like summarize or explain. "
            "Return the title only.\n\n"
            f"Document name: {(document_name or '').strip() or 'Unknown document'}\n"
            f"Content:\n{(text or '')[:1500]}"
        )

        try:
            response = await self._create_chat_completion(
                model=self._deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_completion_tokens=60,
            )

            text_response = ""
            if response.choices and response.choices[0].message:
                text_response = (response.choices[0].message.content or "").strip()

            words = [word for word in text_response.split() if word]
            title = " ".join(words[:5])[:80].strip()
            return title or fallback
        except Exception as e:
            logger.warning("generate_suggestion_title failed: %s", e)
            return fallback

    async def stream_answer(
        self,
        *,
        question: str,
        citations: List[Citation],
        history: Optional[List[dict]] = None,
    ) -> AsyncIterator[str]:
        self._require_configured()

        system_prompt = (
            "You are Infosearch assistant — a helpful, friendly AI. "
            "Handle every type of message naturally:\n"
            "- Casual conversation (greetings, small talk): respond warmly and naturally.\n"
            "- General knowledge (math, facts, definitions): answer directly and concisely.\n"
            "- Document questions: use the provided document excerpts when relevant. "
            "Cite each source inline with its number, e.g. [1] or [2], placed directly after the sentence that uses it. "
            "Only cite numbers that exist in the provided list. "
            "If citations exist but are unrelated, ignore them and answer from general knowledge.\n"
            "- Questions needing private document context not in citations: say you couldn't find relevant information in the documents."
        )

        if citations:
            citations_block = self._citations_context(citations)
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Document excerpts (cite inline with [N] when used — only use numbers from this list):\n{citations_block}\n\n"
                "Answer concisely. Place [N] markers inline after each sentence that draws from a source."
            )
        else:
            user_prompt = f"Message:\n{question}"

        messages = [{"role": "system", "content": system_prompt}]
        for turn in (history or []):
            role = turn.get("role")
            content = turn.get("content", "")
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_prompt})

        stream = await self._create_chat_completion(
            model=self._deployment,
            messages=messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield token
