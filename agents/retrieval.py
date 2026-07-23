import os
from openai import OpenAI
from langsmith.wrappers import wrap_openai
import chromadb
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

load_dotenv()

client = wrap_openai(OpenAI())
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("linkedin_posts")


def _bm25_search(query, all_docs, all_metas, k=10):
    """Sparse keyword search using BM25."""
    corpus = [doc.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [(all_docs[i], all_metas[i], float(scores[i])) for i in top_indices]


def _rrf(dense_results, sparse_results, k=60):
    """Reciprocal Rank Fusion — merge dense and sparse ranked lists."""
    scores = {}
    items = {}

    for ranked_list in [dense_results, sparse_results]:
        for rank, (doc, meta, _) in enumerate(ranked_list, 1):
            key = doc[:200]
            if key not in items:
                items[key] = (doc, meta)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    sorted_keys = sorted(scores, key=scores.get, reverse=True)
    return [(items[key][0], items[key][1]) for key in sorted_keys]


def retrieve(question, n_results=5):
    # Dense: semantic search
    emb = client.embeddings.create(model="text-embedding-3-small", input=question)
    query_embedding = emb.data[0].embedding

    semantic = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results * 2, 20),
        include=["documents", "metadatas", "distances"],
    )
    dense_results = list(zip(
        semantic["documents"][0],
        semantic["metadatas"][0],
        semantic["distances"][0],
    ))

    # Sparse: BM25 over full corpus
    all_data = collection.get(include=["documents", "metadatas"])
    sparse_results = _bm25_search(question, all_data["documents"], all_data["metadatas"], k=n_results * 2)

    # Merge with RRF
    merged = _rrf(dense_results, sparse_results)[:n_results]

    if not merged:
        return "No relevant posts found."

    context = ""
    for i, (doc, meta) in enumerate(merged):
        context += f"\n[{i+1}] {meta.get('author', '?')} — {meta.get('topic', '?')}\n{doc}\n URL: {meta.get('url', '?')}\n"

    prompt = f"""You are a LinkedIn knowledge assistant. Answer the question using only the posts below.

Posts:
{context}

Question: {question}

Answer in detail, include architecture and technical specifics where available, and cite which post you used."""

    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
