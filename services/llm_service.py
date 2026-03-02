from __future__ import annotations

import inspect
import logging
from typing import AsyncIterator, List, Optional

from openai import AsyncAzureOpenAI

from config import Settings
from models.infosearch_api import Citation

logger = logging.getLogger(__name__)


def _is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    text = value.strip()
    return (not text) or ("<" in text and ">" in text)


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

    @staticmethod
    def _sanitize_citations(text: str, citation_count: int) -> str:
        """Strip [N] markers where N is out of range to remove hallucinated refs."""
        import re
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

        response = await self._client.chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
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

        response = await self._client.chat.completions.create(
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

        stream = await self._client.chat.completions.create(
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
