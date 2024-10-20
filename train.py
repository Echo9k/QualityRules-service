# train.py
import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

# Load config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

data_source = config['data_source']
output_index = config['output_index']
output_embeddings = config['output_embeddings']
model_name = config['model_name']

# Load data
df = pd.read_csv(data_source)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

df['embeddings'] = df['text'].apply(lambda x: get_embeddings(x))

# Create FAISS index
embeddings_matrix = np.stack(df['embeddings'].values)
index = faiss.IndexFlatL2(embeddings_matrix.shape[1])
index.add(embeddings_matrix)

# Save FAISS index and DataFrame with embeddings
faiss.write_index(index, output_index)
df.to_pickle(output_embeddings)

print("Training complete. FAISS index and embeddings saved locally.")