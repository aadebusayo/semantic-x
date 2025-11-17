"""
LLM Service for SemanticX Framework.
Provides interface for different LLM providers with retry and simple error handling.
"""
import logging
from typing import Dict, Any, List, Optional
import json
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_llm_config

logger = logging.getLogger(__name__)


class LLMService:
	"""
	Service for interacting with Language Models.
	Supports multiple providers with a unified interface.
	"""
	
	def __init__(self):
		cfg = get_llm_config()
		self.provider = cfg.get("provider", "openai").lower()
		self.model = cfg.get("model", "gpt-4o-mini")
		self.temperature = cfg.get("temperature", 0.1)
		self.max_tokens = cfg.get("max_tokens", 1000)
		self._client = None
		self._init_client(cfg)
		logger.info(f"LLMService initialized: provider={self.provider}, model={self.model}")
	
	def _init_client(self, cfg: Dict[str, Any]):
		if self.provider == "openai":
			try:
				from openai import AsyncOpenAI
				api_key = cfg.get("api_key")
				base_url = cfg.get("base_url")
				self._client = AsyncOpenAI(api_key=api_key, base_url=base_url) if base_url else AsyncOpenAI(api_key=api_key)
			except Exception as e:
				logger.warning(f"OpenAI client not available or API key missing; falling back to mock. {e}")
				self._client = None
		else:
			self._client = None
	
	@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=4), reraise=True)
	async def _openai_chat(
		self,
		messages: List[Dict[str, str]],
		tools: Optional[List[Dict[str, Any]]],
		tool_choice: Optional[str],
		max_tokens: Optional[int],
		temperature: Optional[float],
		model_override: Optional[str] = None,
	) -> Dict[str, Any]:
		from openai import AsyncOpenAI
		assert self._client is not None
		model_name = model_override or self.model
		response = await self._client.chat.completions.create(
			model=model_name,
			messages=messages,
			tools=tools,
			tool_choice=tool_choice,
			max_tokens=max_tokens or self.max_tokens,
			temperature=temperature if temperature is not None else self.temperature,
		)
		choice = response.choices[0]
		result: Dict[str, Any] = {
			"content": choice.message.content if choice.message else "",
		}
		# Convert OpenAI tool calls to our dict shape
		if choice.message and getattr(choice.message, "tool_calls", None):
			converted = []
			for tc in choice.message.tool_calls:
				converted.append({
					"id": tc.id,
					"type": "function",
					"function": {
						"name": tc.function.name,
						"arguments": tc.function.arguments,
					}
				})
			result["tool_calls"] = converted
		return result
	
	async def generate_completion(
		self,
		messages: List[Dict[str, str]],
		tools: Optional[List[Dict[str, Any]]] = None,
		tool_choice: Optional[str] = None,
		max_tokens: Optional[int] = None,
		temperature: Optional[float] = None,
		model_override: Optional[str] = None,
	) -> Dict[str, Any]:
		"""
		Generate a completion from the LLM with retries and provider fallback.
		"""
		try:
			if self.provider == "openai" and self._client is not None:
				return await self._openai_chat(
					messages,
					tools,
					tool_choice,
					max_tokens,
					temperature,
					model_override=model_override,
				)
			# Fallback mock
			logger.info("Using mock LLM response (no provider configured)")
			mock_model = model_override or self.model or "mock-model"
			mock_response = {
				"content": f"This is a mock response from {mock_model}. Configure OPENAI_API_KEY to enable real completions.",
			}
			if tools and tool_choice == "auto":
				mock_response["tool_calls"] = [{
					"id": "mock_tool_call_1",
					"type": "function",
					"function": {
						"name": tools[0].get('function', {}).get('name', 'mock_tool'),
						"arguments": json.dumps({"input": "mock_input"})
					}
				}]
			return mock_response
		except Exception as e:
			logger.error(f"LLM completion failed: {e}", exc_info=True)
			return {"content": f"Error generating response: {str(e)}", "error": True}
	
	async def test_connection(self) -> bool:
		try:
			_ = await self.generate_completion(messages=[{"role": "user", "content": "ping"}])
			return True
		except Exception:
			return False
