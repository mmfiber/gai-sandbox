from sentence_transformers import SentenceTransformer
import json
import pandas as pd
import numpy as np
from qdrant_client.models import VectorParams, Distance, Filter
from gai_sandbox.qdrant import get_client
from fastapi import FastAPI
from pathlib import Path

tf = SentenceTransformer('all-MiniLM-L6-v2', device="cpu") # or device="cpu" if you don't have a GPU

# df = pd.read_json(Path(__file__).parent / 'startups_demo.jsonl', lines=True)
# vectors = tf.encode([
#     row.alt + ". " + row.description
#     for row in df.itertuples()
# ], show_progress_bar=True)
# np.save(Path(__file__).parent / 'startup_vectors.npy', vectors, allow_pickle=False)
vectors = np.load(Path(__file__).parent / 'startup_vectors.npy')
print(vectors.shape)

# # payload is now an iterator over startup data
# fd = open(Path(__file__).parent / 'startups_demo.jsonl')
# payload = map(json.loads, fd)

client = get_client()
# client.recreate_collection(
#     collection_name='startups', 
#     vectors_config=VectorParams(
#         size=tf.get_sentence_embedding_dimension(),
#         distance=Distance.COSINE
#     ),
# )
# client.upload_collection(
#     collection_name='startups',
#     vectors=vectors,
#     payload=payload,
#     ids=None,  # Vector ids will be assigned automatically
#     batch_size=256  # How many vectors will be uploaded in a single request?
# )

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.tf = tf
        # initialize Qdrant client
        self.client = client

    def search(self, text: str, query_filter: Filter | None = None):
        # Convert text query into vector
        vector = self.tf.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=query_filter,  # If you don't want any filters for now
            limit=5  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads


app = FastAPI()

# Create a neural searcher instance
neural_searcher = NeuralSearcher(collection_name='startups')

@app.get("/api/search")

def search_startup(text: str, city: str | None = None):
    city_fliter = None if city is None else Filter(must=[{
        "key": "city", # Store city information in a field of the same name 
        "match": { # This condition checks if payload field has the requested value
            "value": city
        }
    }])
    return {
        "result": neural_searcher.search(text=text, query_filter=city_fliter)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
