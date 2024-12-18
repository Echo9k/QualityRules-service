import streamlit as st
import pandas as pd
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

# Set paths
path_root = Path.cwd()
while not (path_root / "data").exists():
    if path_root == path_root.parent:
        raise FileNotFoundError("Directory 'data' not found")
    path_root = path_root.parent

path_data = path_root / "data"
model_path = path_root / "sentence-transformers/all-MiniLM-L6-v2"

# Load data
data_BoR = pd.read_excel(
    path_data / 'processed/Bank_of_Rules-refractored.xlsx',
    usecols=['Description', 'RuleID', 'Code', 'Parameters'],
    index_col='RuleID'
)

df_defs = (
    pd.read_json(path_data / 'internim/definitions.json')[['RuleID', 'Description']]
    .drop_duplicates(subset=['RuleID'])
    .set_index('RuleID')
)

# Load tokenizer and model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path / "tokenizer")
    model = AutoModel.from_pretrained(model_path / "model")
    return tokenizer, model

tokenizer, model = load_model()

# Load FAISS index
@st.cache_resource
def load_faiss_index():
    index = faiss.read_index(str(path_data / "processed/faiss_index.idx"))
    return index

index = load_faiss_index()

# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Search function
def search(query, top_k=5):
    query_embedding = get_embeddings(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = df_defs.iloc[indices[0]]
    return results, distances[0]

# Streamlit UI
st.title("Rule Search Application")
st.write("Search for rules by providing a query.")

query = st.text_input("Enter your query:", "")
top_k = st.slider("Number of results to display:", min_value=1, max_value=10, value=5)

if st.button("Search"):
    if query.strip():
        st.write("## Results")
        try:
            results, distances = search(query, top_k=top_k)
            for i, (idx, row) in enumerate(results.iterrows()):
                st.write(f"### Rank {i+1}")
                st.write(f"**RuleID**: {idx}")
                st.write(f"**Description**: {row['Description']}")
                st.write(f"**Distance**: {distances[i]:.4f}")
                st.write("---")
        except Exception as e:
            st.error(f"Error during search: {str(e)}")
    else:
        st.warning("Please enter a query before searching.")
