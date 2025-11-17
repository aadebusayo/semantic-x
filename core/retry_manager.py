"""
Retry and escalation controller for SemanticX Framework.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetryDecision:
	should_retry: bool
	escalate_model: bool = False
	new_model_tier: Optional[str] = None
	message: Optional[str] = None


class RetryManager:
	"""Centralized retry policy for tool execution."""

	def __init__(self):
		self.max_attempts = getattr(settings, "max_retry_attempts", 3)
		self.escalate_after = getattr(settings, "retry_escalate_attempts", max(1, self.max_attempts - 1))
		self.retry_delay = getattr(settings, "retry_delay_seconds", 1.0)
		self.fallback_tier = "fallback"

	def reset(self, state):
		state.retry_count = 0

	def handle_failure(self, state, function_name: str, retryable: bool) -> RetryDecision:
		state.retry_count += 1
		logger.warning(
			"Tool failure: session=%s tool=%s retry_count=%s retryable=%s",
			state.session_id,
			function_name,
			state.retry_count,
			retryable,
		)

		needs_auth_refresh = state.auth_pending and state.retry_count == 1
		escalate = not needs_auth_refresh and retryable and state.retry_count >= self.escalate_after
		should_retry = retryable and state.retry_count < self.max_attempts

		new_tier = self.fallback_tier if escalate else None
		message = None
		if not should_retry and retryable:
			message = "Maximum retry attempts reached"
		elif not retryable:
			message = "Failure marked non-retryable"

		if needs_auth_refresh:
			return RetryDecision(should_retry=True, message="auth_refresh_required")

		return RetryDecision(
			should_retry=should_retry,
			escalate_model=escalate,
			new_model_tier=new_tier,
			message=message,
		)

