from fastapi import APIRouter
import numpy as np

router = APIRouter()

def setup(model, index, documents, membership, cache):

    @router.post("/query")
    def query_api(payload: dict):

        query = payload["query"]

        query_embedding = model.encode(query)

        hit, matched_query, result, score = cache.lookup(query_embedding)

        if hit:

            return {
                "query": query,
                "cache_hit": True,
                "matched_query": matched_query,
                "similarity_score": float(score),
                "result": result
            }

        distances, indices = index.search(
            np.array([query_embedding]), 5
        )

        doc_id = indices[0][0]

        result = documents[doc_id]

        cluster = int(np.argmax(membership[doc_id]))

        cache.store(query, query_embedding, result, cluster)

        return {
            "query": query,
            "cache_hit": False,
            "matched_query": None,
            "similarity_score": float(score),
            "result": result,
            "dominant_cluster": cluster
        }


    @router.get("/cache/stats")
    def cache_stats():
        return cache.stats()


    @router.delete("/cache")
    def clear_cache():
        cache.clear()
        return {"message": "Cache cleared"}