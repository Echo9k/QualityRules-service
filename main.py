from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

# Load config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load FAISS index and embeddings DataFrame
output_index = config['output_index']
output_embeddings = config['output_embeddings']
model_name = config['model_name']

# Load the FAISS index
index = faiss.read_index(output_index)

# Load the DataFrame with embeddings
df = pd.read_pickle(output_embeddings)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

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
def query_faiss(query: str, top_k: int = 5, verbosity: int = 1):
    try:
        if verbosity > 0:
            print("✅# Generate embeddings for the query text")
        query_embedding = get_embeddings(query)
        
        if verbosity > 0:
            print("🔍# Search the FAISS index")
        # Perform FAISS search for nearest neighbors
        distances, indices = index.search(np.array([query_embedding]), top_k)
        
        # Get the top_k most similar items
        if verbosity > 0:
            print("🎉# Return the results")

        # Add the distances directly to the results DataFrame, then convert to a dictionary
        results_df = df.iloc[indices[0]].drop(columns=["embeddings"]).copy()
        results_df['distance'] = distances[0]  # Directly assign the distances array

        # Convert the DataFrame to a list of dictionaries
        results = results_df.to_dict(orient='records')
        
        if verbosity > 0:
            print("🚀# Done!")
        if verbosity > 1:
            print(f"🔥# Results: {results}")
        
        return {
            "query": query,
            "results": results
        }
    
    except Exception as e:
        if verbosity >= 1:
            print(f"🛑 An error occurred: {str(e)}")
        # You could add more detailed error messages or logging here for verbosity level 3
        if verbosity >= 3:
            print("Debugging information:", e)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "FastAPI app is running!"}
