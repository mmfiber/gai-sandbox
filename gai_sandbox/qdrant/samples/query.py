from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from gai_sandbox.qdrant import get_client

client = get_client()
# search_result = client.search(
#     collection_name="test_collection",
#     query_vector=[0.2, 0.1, 0.9, 0.7], 
#     limit=3
# )
search_result = client.search(
    collection_name="test_collection",
    query_vector=[0.2, 0.1, 0.9, 0.7], 
    query_filter=Filter(
        must=[
            FieldCondition(
                key="city",
                match=MatchValue(value="London")
            )
        ]
    ),
    limit=3
)
print(search_result)

