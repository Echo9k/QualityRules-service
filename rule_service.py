# %%
from pathlib import Path


path_root = Path.cwd()
while not (path_root / "data").exists():
    if path_root == path_root.parent:
        raise FileNotFoundError("Directory 'data' not found")
    path_root = path_root.parent

path_data = path_root / "data"

# %% [markdown]
# ### 1. **Backend Service**: FastAPI
# 
# We'll use FastAPI to serve your model and provide an endpoint for querying.
# 
# #### Steps:
# 
# 1. **Set up FastAPI**.
# 2. **Load your existing FAISS index and embeddings**.
# 3. **Serve a POST endpoint** to handle user queries and return the most similar rules.

# %%
from fastapi import FastAPI, Query
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Initialize FastAPI
app = FastAPI(title="Rule Search Service", description="A local API to search for the most adequate rules.")

# %%
# Paths
data_path = path_root / "data"
model_path = path_root / "sentence-transformers/all-MiniLM-L6-v2"

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
model = AutoModel.from_pretrained(model_path / "model")

# Load rule definitions and FAISS index
df_defs = pd.read_json(data_path / 'internim/definitions.json')[['RuleID', 'Description']]
index = faiss.read_index(str(data_path / "processed/faiss_index.idx"))  # Prebuilt index file

# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Search function
def search_faiss(query: str, top_k: int = 5):
    query_embedding = get_embeddings(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = df_defs.iloc[indices[0]]
    return results, distances[0]

# API Input Schema
class QueryInput(BaseModel):
    query: str
    top_k: int = 5

# Endpoint for querying
@app.post("/search/")
def search_rules(input: QueryInput):
    results, distances = search_faiss(input.query, input.top_k)
    response = []
    for i, row in results.iterrows():
        response.append({
            "RuleID": row["RuleID"],
            "Description": row["Description"],
            "Distance": float(distances[i])
        })
    return {"results": response}


# %%



