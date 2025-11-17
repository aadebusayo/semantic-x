"""
Telemetry utilities for capturing routing/LLM/tool events.
"""
from __future__ import annotations

import logging
from typing import Dict, Any
from datetime import datetime, timezone

from config import settings

logger = logging.getLogger(__name__)


def _build_event(event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
	return {
		"type": event_type,
		"timestamp": datetime.now(timezone.utc).isoformat(),
		**payload,
	}


def record_event(state, event_type: str, payload: Dict[str, Any]):
	if not getattr(settings, "enable_observability", True):
		return
	event = _build_event(event_type, payload)
	state.telemetry_events.append(event)
	logger.debug("Telemetry event captured: %s", event)

