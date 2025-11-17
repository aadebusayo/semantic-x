"""
Schema validation helpers for tool execution.

Uses JSON Schema to validate tool arguments before calling downstream APIs.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

try:
	from jsonschema import Draft202012Validator, ValidationError
except Exception:  # pragma: no cover - handled at runtime
	Draft202012Validator = None
	ValidationError = None

logger = logging.getLogger(__name__)


class SchemaValidator:
	"""JSON schema validator with caching."""

	def __init__(self):
		self._validator_cache: Dict[str, Any] = {}

	def validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str]]:
		"""
		Validate tool arguments against the provided schema.

		Returns:
			(validated_data, error_message)
		"""
		if not schema:
			return data or {}, None

		if not Draft202012Validator:
			logger.warning("jsonschema not available; skipping validation")
			return data or {}, None

		try:
			validator = self._get_validator(schema)
			payload = data or {}
			validator.validate(payload)
			return payload, None
		except ValidationError as exc:
			path = ".".join([str(p) for p in exc.path])
			error = f"Invalid value for '{path or exc.schema_path[-1]}': {exc.message}"
			logger.debug("Schema validation failed: %s", error)
			return {}, error
		except Exception as exc:  # pragma: no cover
			logger.error("Schema validation error: %s", exc, exc_info=True)
			return {}, "Unexpected validation error"

	def _get_validator(self, schema: Dict[str, Any]):
		key = str(hash(str(schema)))
		if key not in self._validator_cache:
			self._validator_cache[key] = Draft202012Validator(schema)
		return self._validator_cache[key]

