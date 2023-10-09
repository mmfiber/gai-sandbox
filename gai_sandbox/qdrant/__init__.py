from qdrant_client import QdrantClient

_client: QdrantClient | None = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
      _client = QdrantClient("localhost", port=6333)
    return _client
