"""
Vector Store service for SemanticX Framework.
Supports Qdrant (open-source) with in-memory fallback.
"""
from typing import List, Dict, Any, Optional
from ..config import get_vector_store_config

try:
	from qdrant_client import QdrantClient
	except_import_error = None
except Exception as e:
	except_import_error = e
	QdrantClient = None  # type: ignore


class InMemoryVectorStore:
	def __init__(self, dimension: int = 1536):
		self.dimension = dimension
		self.vectors: List[Dict[str, Any]] = []
	
	def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any]):
		self.vectors = [v for v in self.vectors if v.get("id") != id]
		self.vectors.append({"id": id, "vector": vector, "metadata": metadata})
	
	def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
		# Dummy similarity: return last items
		return self.vectors[-top_k:]


class QdrantVectorStore:
	def __init__(self, url: str, api_key: Optional[str], collection: str = "semanticx", dimension: int = 1536):
		if QdrantClient is None:
			raise RuntimeError(f"qdrant-client is not available: {except_import_error}")
		self.client = QdrantClient(url=url, api_key=api_key) if api_key else QdrantClient(url=url)
		self.collection = collection
		self.dimension = dimension
		self._ensure_collection()
	
	def _ensure_collection(self):
		from qdrant_client.http import models as rest
		try:
			self.client.get_collection(self.collection)
		except Exception:
			self.client.recreate_collection(
				collection_name=self.collection,
				vectors_config=rest.VectorParams(size=self.dimension, distance=rest.Distance.COSINE)
			)
	
	def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any]):
		from qdrant_client.http import models as rest
		self.client.upsert(
			collection_name=self.collection,
			points=[rest.PointStruct(id=id, vector=vector, payload=metadata)],
		)
	
	def query(self, vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
		from qdrant_client.http import models as rest
		res = self.client.search(
			collection_name=self.collection,
			query_vector=vector,
			limit=top_k,
		)
		return [
			{"id": p.id, "score": p.score, "metadata": p.payload} for p in res
		]


def get_vector_store():
	config = get_vector_store_config()
	type_ = config.get("type", "memory").lower()
	if type_ == "qdrant" and config.get("url"):
		return QdrantVectorStore(
			url=config.get("url"),
			api_key=config.get("api_key"),
			collection=config.get("collection", "semanticx"),
			dimension=config.get("dimension", 1536)
		)
	# Fallback in-memory
	return InMemoryVectorStore(dimension=config.get("dimension", 1536))
