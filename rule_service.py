from pathlib import Path
from fastapi import FastAPI, Query
from pydantic import BaseModel
import faiss
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import logging


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Rule Search Service",
    description="A local API to search for the most adequate rules.",
    version="1.0.0"
)

# Paths
path_root = Path.cwd()
while not (path_root / "data").exists():
    if path_root == path_root.parent:
        raise FileNotFoundError("Directory 'data' not found")
    path_root = path_root.parent

path_data = path_root / "data"
model_path = path_root / "sentence-transformers/all-MiniLM-L6-v2"

# Load model and tokenizer
logger.info("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
model = AutoModel.from_pretrained(model_path / "model")

# Load rule definitions and FAISS index
logger.info("Loading rule definitions and FAISS index...")
df_defs = pd.read_json(path_data / 'internim/definitions.json')[['RuleID', 'Description']]
index = faiss.read_index(str(path_data / "processed/faiss_index.idx"))

# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Search function
def search_faiss(query: str, top_k: int = 5):
    logger.info("Performing FAISS search...")
    query_embedding = get_embeddings(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = df_defs.iloc[indices[0]]
    return results, distances[0]

# API Input Schema
class QueryInput(BaseModel):
    query: str
    top_k: int = 5

# Health Check Endpoint
@app.get("/health", include_in_schema=False)
def health_check():
    """Health check endpoint to verify service status."""
    logger.info("Health check called.")
    return {"status": "ok", "message": "Service is running."}

# Search Endpoint
@app.post("/search/")
def search_rules(input: QueryInput):
    logger.info(f"Search query received: {input.query}")
    results, distances = search_faiss(input.query, input.top_k)
    response = []
    for i, row in results.iterrows():
        response.append({
            "RuleID": row["RuleID"],
            "Description": row["Description"],
            "Distance": float(distances[i])
        })
    logger.info("Search completed successfully.")
    return {"results": response}

# Instructions for running with latest versions
# Install FastAPI and Uvicorn if not already installed:
# pip install fastapi uvicorn --upgrade

# To start the service:
# uvicorn <script_name>:app --reload --host 127.0.0.1 --port 8000