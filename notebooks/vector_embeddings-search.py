# %% [markdown]
# ## 1. Set up the environment
# First, ensure you have the necessary libraries installed for embedding generation, vector search, and data manipulation. You can use models from transformers for embedding generation, faiss for similarity search, and pandas for working with your DataFrame.

# %%
from pathlib import Path


path_root = Path.cwd()
while not (path_root / "data").exists():
    if path_root == path_root.parent:
        raise FileNotFoundError("Directory 'data' not found")
    path_root = path_root.parent

path_data = path_root / "data"

# %%
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import re

from sentence_transformers import SentenceTransformer

# %% [markdown]
# ## 2. Prepare your DataFrame
# You should have a DataFrame that contains a text column for which you want to perform a similarity search.

# %%
data_BoR = pd.read_excel(
    path_root / 'data/processed/Bank_of_Rules-refractored.xlsx',
    usecols=['Description', 'RuleID', 'Code', 'Parameters'],
    index_col='RuleID'
    )

df_defs = (pd.read_json(path_root / 'data/internim/definitions.json')[['RuleID', 'Description',
                                   'Condition/Logic', 'Example',
                                #    'Parameters',
                                   'Category', 'Categorization']]
            .drop_duplicates(subset=['RuleID'])).set_index('RuleID')

# %%
display(df_defs.columns)
df_defs = (
    df_defs
    .drop_duplicates()
    # .shape
    )
df_defs

# %% [markdown]
# ## 3. Generate Vector Embeddings
# Use a pre-trained transformer model like distilbert-base-uncased or sentence-transformers to generate embeddings for each entry in the text column.*italicized text*

# %%
from transformers import AutoTokenizer, AutoModel
import torch

# Load a transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Save your files to a specified directory with PreTrainedModel.save_pretrained():
tokenizer.save_pretrained(path_root / "sentence-transformers/all-MiniLM-L6-v2/tokenizer")
model.save_pretrained(path_root / "sentence-transformers/all-MiniLM-L6-v2/model")

# #Now when you’re offline, reload your files with PreTrainedModel.from_pretrained() from the specified directory:
# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2/tokenizer")
# model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2/model")

# %%
# Function to generate embeddings
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate embeddings for each row
df_defs['embeddings'] = df_defs['Description'].apply(lambda x: get_embeddings(x))

# %% [markdown]
# ## 4. Build the Search Engine with FAISS
# You can now use faiss to index these embeddings and perform similarity searches.

# %%
import faiss
import numpy as np

# Prepare the embeddings matrix
embeddings_matrix = np.stack(df_defs['embeddings'].values)

# Build the FAISS index
index = faiss.IndexFlatL2(embeddings_matrix.shape[1])  # Using L2 distance
index.add(embeddings_matrix)

# Search function to find the most similar texts
def search(query, top_k=5):
    query_embedding = get_embeddings(query).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    results = df_defs.iloc[indices[0]]
    return results, distances[0]

# %%
query = "CDE should be unique"
results, distances = search(query, top_k=4)

# Display the most similar texts
print("Results:")
print(results[['RuleID','Description']])
print("Distances:", distances)

# %% [markdown]
# ## 5. Perform a Similarity Search
# You can now search your DataFrame for similar text using the search function.

# %%
row

# %%


# %%
from IPython.display import Markdown, display

# Print out the most similar texts
display(Markdown("### Results:"))
for i, rule_id in enumerate(results.index):
    row = pd.concat([df_defs.loc[rule_id], data_BoR.loc[rule_id]])
    display(Markdown(f"#### RuleID: {rule_id}"))
    for i in row.items():
        print(f"\033[1m{i[0]}:\033[0m {i[1]}")
        

# %%



