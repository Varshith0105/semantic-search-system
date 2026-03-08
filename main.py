from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

from Cache.semantic_cache import SemanticCache
from api.routes import router, setup


app = FastAPI(
    title="Semantic Search API",
    description="Semantic search system with fuzzy clustering and semantic cache",
    version="1.0"
)

# Redirect root URL to Swagger UI
@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs")


print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")


print("Loading vector index...")

index = faiss.read_index("models/vector_index.faiss")

embeddings = np.load("models/embeddings.npy")

membership = np.load("models/cluster_memberships.npy")


print("Loading dataset...")

documents = []

data_path = "data/20_newsgroups"

for category in os.listdir(data_path):

    folder = os.path.join(data_path, category)

    for file in os.listdir(folder):

        file_path = os.path.join(folder, file)

        try:
            with open(file_path, "r", encoding="latin1") as f:
                documents.append(f.read())
        except:
            pass


print("Initializing semantic cache...")

cache = SemanticCache()


print("Setting up API routes...")

setup(model, index, documents, membership, cache)

app.include_router(router)


print("Server ready!")