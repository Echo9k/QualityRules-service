from starlette.requests import Request
import ray
from ray import serve
from fastapi import FastAPI
import pandas as pd
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

# Load serve configuration
with open('serve.yaml', 'r') as f:
    config = yaml.safe_load(f)

index_path = config['index_path']
embeddings_path = config['embeddings_path']
port = config['port']
replicas = config['replicas']

# Shutdown any previous Ray instances before initializing a new one
if ray.is_initialized():
    ray.shutdown()

# Initialize Ray Serve
ray.init(ignore_reinit_error=True)

# Create FastAPI app
app = FastAPI()

# Load FAISS index and embeddings
df = pd.read_pickle(embeddings_path)
index = faiss.read_index(index_path)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Function to generate embeddings for query
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Define the Ray Serve deployment
@serve.deployment(num_replicas=replicas, ray_actor_options={"num_cpus": 0.01, "num_gpus": 0})
class SearchEngine:
    def __init__(self):
        self.df = df
        self.index = index

    def search(self, query, top_k=5):
        query_embedding = get_embeddings(query).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)
        results = self.df.iloc[indices[0]]
        return results[['id', 'text']].to_dict(orient="records"), distances[0].tolist()

    async def __call__(self, request: Request):
        body = await request.json()
        query_text = body.get('query')
        top_k = body.get('top_k', 5)
        results, distances = self.search(query_text, top_k)
        return {"results": results, "distances": distances}

# Bind the FastAPI app to the deployment
search_engine_app = SearchEngine.bind()

# Start the FastAPI server via Ray Serve
if __name__ == "__main__":
    serve.run(search_engine_app)
