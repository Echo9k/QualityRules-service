from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load FAISS index and embeddings DataFrame
output_index = config['output_index']
output_embeddings = config['output_embeddings']
model_name = config['model_name']

# Load the FAISS index
try:
    index = faiss.read_index(output_index)
    logger.info("FAISS index loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load FAISS index: {str(e)}")
    raise RuntimeError("Could not load FAISS index.")

# Load the DataFrame with embeddings
try:
    df = pd.read_pickle(output_embeddings)
    logger.info("Embeddings DataFrame loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embeddings DataFrame: {str(e)}")
    raise RuntimeError("Could not load embeddings DataFrame.")

# Load the tokenizer and model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    logger.info("Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Transformer model: {str(e)}")
    raise RuntimeError("Could not load Transformer model.")

# FastAPI app
app = FastAPI()

# Function to generate embeddings for new queries
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Endpoint to query similar items from the FAISS index
@app.get("/query/")
def query_faiss(query: str = Query(..., min_length=1, description="The query text for similarity search"),
                top_k: int = Query(5, ge=1, le=100, description="Number of top similar items to retrieve"),
                verbosity: int = Query(1, ge=0, le=3, description="Verbosity level for logging")):
    start_time = time.time()
    
    try:
        # Logging query and verbosity level
        if verbosity > 0:
            logger.info(f"Received query: '{query}' with top_k={top_k} and verbosity={verbosity}")

        # Generate embeddings for the query
        query_embedding = get_embeddings(query)
        
        # Search the FAISS index for the nearest neighbors
        distances, indices = index.search(np.array([query_embedding]), top_k)
        
        # Get the top_k most similar items
        results_df = df.iloc[indices[0]].drop(columns=["embeddings"]).copy()
        results_df['distance'] = distances[0]

        # Convert the DataFrame to a list of dictionaries
        results = results_df.to_dict(orient='records')
        
        # Log the results if verbosity is high
        if verbosity > 1:
            logger.debug(f"Results: {results}")

        # Track and log response time
        response_time = time.time() - start_time
        if verbosity > 0:
            logger.info(f"Query processed in {response_time:.2f} seconds.")

        return {
            "query": query,
            "response_time": response_time,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        if verbosity >= 3:
            logger.debug("Detailed error information:", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI app is running!"}
