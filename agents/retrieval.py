import os
from openai import OpenAI
import chromadb
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("linkedin_posts")


def retrieve(question):
    emb = client.embeddings.create(model="text-embedding-3-small", input=question)
    query_embedding = emb.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "No relevant posts found."

    context = ""
    for i, (doc, meta) in enumerate(zip(docs, metas)):
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
