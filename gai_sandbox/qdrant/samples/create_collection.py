from qdrant_client.http.models import Distance, VectorParams
from gai_sandbox.qdrant import get_client

client = get_client()
client.recreate_collection(
    collection_name="test_collection",
    vectors_config=VectorParams(size=4, distance=Distance.DOT),
)
