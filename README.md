# Semantic Search System with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a **semantic search system** built using modern AI/ML techniques.  
The system allows users to search a large corpus of documents based on **meaning rather than exact keyword matching**.

Traditional search systems rely on exact keywords.

Example:

Query 1
```
space shuttle launch
```

Query 2
```
NASA rocket launch
```

Even though these queries have similar meaning, traditional search engines treat them as different queries.

This system solves that problem using:

- Sentence embeddings
- Vector similarity search
- Fuzzy clustering
- Semantic query caching
- FastAPI service

The project demonstrates how modern **AI-powered search systems** work internally.

---

# System Architecture

```
User Query
    │
    ▼
FastAPI API
    │
    ▼
Sentence Transformer (Embedding Model)
    │
    ▼
Semantic Cache Check
    │
    ├── Cache Hit → Return cached result
    │
    ▼
Vector Database (FAISS)
    │
    ▼
Retrieve Similar Documents
    │
    ▼
Fuzzy Clustering
    │
    ▼
Store Result in Cache
    │
    ▼
Return Response
```

The system efficiently processes queries by combining **semantic similarity search with intelligent caching**.

---

# Dataset

The system uses the **20 Newsgroups dataset**, a well-known benchmark dataset for text classification and clustering.

Dataset properties:

- ~20,000 documents
- 20 topic categories
- Usenet discussion posts

Example categories:

```
alt.atheism
comp.graphics
comp.os.ms-windows.misc
comp.sys.ibm.pc.hardware
comp.sys.mac.hardware
comp.windows.x
misc.forsale
```

Each document is treated as a searchable semantic unit.

---

# Key Components

## 1. Embedding Model

The system uses the **Sentence Transformers model**

```
all-MiniLM-L6-v2
```

Reasons for choosing this model:

- Lightweight
- Fast inference
- High semantic accuracy
- Suitable for real-time applications

Each document is converted into a **384-dimensional vector representation**.

---

## 2. Vector Database

Document embeddings are stored using:

**FAISS (Facebook AI Similarity Search)**

FAISS allows efficient similarity search in high-dimensional vector spaces.

Advantages:

- Extremely fast vector search
- Optimized for large datasets
- Low memory overhead

---

## 3. Fuzzy Clustering

Unlike traditional clustering methods where each document belongs to only one cluster, this system uses **Fuzzy C-Means clustering**.

This allows documents to belong to **multiple clusters with different probabilities**.

Example:

```
Document A
Politics: 0.55
Firearms: 0.30
Religion: 0.15
```

This better represents **real-world semantic overlap between topics**.

---

## 4. Semantic Cache

Traditional caching works only for **exact queries**.

Example:

```
Query: space shuttle launch
```

But users may ask the same question differently:

```
NASA rocket launch
launch of NASA shuttle
space mission launch
```

The system compares **query embeddings using cosine similarity**.

If similarity exceeds a threshold:

```
Cache Hit
```

Otherwise:

```
Cache Miss
```

This significantly improves system performance.

---

# API Implementation

The API is implemented using **FastAPI**, a high-performance Python web framework.

The server loads the following components at startup:

- Sentence transformer model
- FAISS vector index
- Cluster membership data
- Dataset documents

---

# API Endpoints

## 1. Semantic Search

Endpoint

```
POST /query
```

Example request

```json
{
 "query": "space shuttle launch"
}
```

Example response

```json
{
 "query": "space shuttle launch",
 "cache_hit": false,
 "matched_query": null,
 "similarity_score": 0.0,
 "result": "Retrieved document text...",
 "dominant_cluster": 3
}
```

---

## 2. Cache Statistics

Endpoint

```
GET /cache/stats
```

Example response

```json
{
 "total_entries": 5,
 "hit_count": 3,
 "miss_count": 2,
 "hit_rate": 0.6
}
```

---

## 3. Clear Cache

Endpoint

```
DELETE /cache
```

Response

```
Cache cleared successfully
```

---

# Project Structure

```
semantic-search-system
│
├── api
│   └── routes.py
│
├── Cache
│   └── semantic_cache.py
│
├── data
│   └── 20_newsgroups
│
├── models
│   ├── vector_index.faiss
│   ├── embeddings.npy
│   └── cluster_memberships.npy
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```
git clone https://github.com/Varshith0105/semantic-search-system.git
cd semantic-search-system
```

Create a virtual environment

```
python -m venv venv
```

Activate the environment

Windows

```
venv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Running the Server

Start the API server

```
uvicorn main:app --reload
```

Server runs at

```
http://127.0.0.1:8000
```

Opening this URL will automatically redirect to

```
http://127.0.0.1:8000/docs
```

This opens the **Swagger UI**, where you can test the API.

---

# Example Query

Example request

```json
{
 "query": "NASA rocket launch"
}
```

The system will:

1. Convert query into embedding  
2. Check semantic cache  
3. Search FAISS vector index  
4. Retrieve similar document  
5. Identify dominant cluster  
6. Return response  

---

# Technologies Used

- Python
- FastAPI
- Sentence Transformers
- FAISS
- NumPy
- Scikit-learn

---

# Future Improvements

Potential improvements include:

- Redis-based distributed caching
- GPU acceleration for embeddings
- Incremental indexing
- Docker containerization
- Cloud deployment
- Web-based search interface

---

# Author

**Varshith Julakanti**

GitHub

```
https://github.com/Varshith0105
```

---

# License

This project is created for educational and demonstration purposes.
