"""
Generic HTTP tool executor for tools loaded from OpenAPI schemas.
"""
from typing import Dict, Any, Optional
import httpx


class HttpToolExecutor:
	def __init__(self, timeout: float = 30.0):
		self.timeout = timeout
	
	def _build_url(self, servers: list, path: str) -> str:
		base = servers[0].get("url") if servers else ""
		if base and base.endswith("/") and path.startswith("/"):
			return base[:-1] + path
		return (base or "") + path
	
	async def execute(self, http_meta: Dict[str, Any], arguments: Dict[str, Any]) -> Dict[str, Any]:
		method = http_meta.get("method", "GET").upper()
		path = http_meta.get("path", "/")
		servers = http_meta.get("servers", [])
		url = self._build_url(servers, path)
		
		params = {}
		json_body = None
		# naive split: treat simple types as query, nested as body
		for k, v in (arguments or {}).items():
			if isinstance(v, (str, int, float, bool)):
				params[k] = v
			else:
				json_body = json_body or {}
				json_body[k] = v
		
		async with httpx.AsyncClient(timeout=self.timeout) as client:
			resp = await client.request(method, url, params=params, json=json_body)
			resp.raise_for_status()
			try:
				data = resp.json()
			except Exception:
				data = {"text": resp.text}
			return {"status": resp.status_code, "data": data}
