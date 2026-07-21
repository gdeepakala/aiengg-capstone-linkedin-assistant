import os
import json
import requests
from openai import OpenAI
from pydantic import BaseModel, ValidationError
import chromadb

from agents.resources import (
    extract_links,
    fetch_resource,
    is_shortlink,
    is_supported_resource,
    resolve_redirect,
)

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("linkedin_posts")

# how much of each fetched resource to keep, and how big a stored document can get
PER_RESOURCE_CHARS = 4000
MAX_CONTENT_CHARS = 8000
MAX_EMBED_CHARS = 8000


class Resource(BaseModel):
    author: str
    topic: str
    community: str
    summary: str
    tags: list[str]
    url: str
    date: str


def _search(query):
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": os.getenv("SERPER_API_KEY"),
            "Content-Type": "application/json",
        },
        json={"q": query, "num": 10},
    )
    resp.raise_for_status()
    # serper's "organic" results already use title / snippet / link
    return resp.json().get("organic", [])


def _gather_resources(item):
    """Follow *useful* links in the title/snippet/slug and return their text.

    Skips linkedin.com (login-walled — yields only boilerplate) and only reads
    link types we can reliably parse. lnkd.in-style shortlinks are resolved to
    their real destination first, in case they point at a repo/paper/video.
    """
    haystack = f"{item.get('title', '')} {item.get('snippet', '')} {item.get('link', '')}"
    blobs = []
    for url in extract_links(haystack):
        if is_shortlink(url):
            url = resolve_redirect(url)
        if "linkedin.com" in url.lower() or not is_supported_resource(url):
            continue
        kind, text = fetch_resource(url)
        if text.strip():
            blobs.append(f"[{kind}] {url}\n{text.strip()[:PER_RESOURCE_CHARS]}")
            print(f"    + read {kind}: {url[:70]}")
    return "\n\n".join(blobs)[:MAX_CONTENT_CHARS]


def _extract(item, resource_text=""):
    prompt = f"""Extract metadata from this LinkedIn post search result. Return JSON only.

Title: {item.get("title", "")}
Snippet: {item.get("snippet", "")}
URL: {item.get("link", "")}

Linked resource content (README / transcript / paper / PDF — may be empty):
{resource_text[:6000]}

Fields:
- author: who wrote/owns it (a person's name if visible; else "Unknown")
- topic: main topic in 1-5 words
- community: "aiengg" or "ebpf" based on content
- summary: 3-5 sentences. Use the linked resource content when present to
  describe what the project/paper/video actually does. Keep concrete names.
- tags: list of 3-5 tags
- url: the URL above
- date: YYYY-MM-DD if visible, else "unknown"
"""
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Resource(**data)


def _store(resource, item, resource_text=""):
    # the stored document keeps real text (raw snippet + resource excerpt), not
    # just the lossy summary, so names/details survive into retrieval.
    parts = [resource.summary, item.get("title", ""), item.get("snippet", "")]
    if resource_text:
        parts.append(resource_text)
    content = "\n".join(p for p in parts if p).strip()[:MAX_CONTENT_CHARS]

    emb_input = (
        f"{resource.author} {resource.topic} {' '.join(resource.tags)}\n{content}"
    )[:MAX_EMBED_CHARS]
    emb = client.embeddings.create(model="text-embedding-3-small", input=emb_input)
    embedding = emb.data[0].embedding

    metadata = resource.model_dump()
    metadata["tags"] = ", ".join(resource.tags)

    doc_id = resource.url[-90:].replace("/", "_").replace(":", "_")
    collection.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content],
        metadatas=[metadata],
    )
    print(f"  stored: {resource.author} — {resource.topic}")


def ingest(query):
    print(f"searching: {query}")
    items = _search(query)
    print(f"found {len(items)} results")
    for item in items:
        try:
            resource_text = _gather_resources(item)
            resource = _extract(item, resource_text)
            _store(resource, item, resource_text)
        except (ValidationError, Exception) as e:
            print(f"  skipped {item.get('link', '?')[:60]}: {e}")
    print("done.")


def ingest_resource(url_or_path):
    """Manually ingest a single resource: a URL (GitHub/YouTube/arXiv/PDF/web)
    or a local file path (e.g. an attached PDF)."""
    print(f"reading: {url_or_path}")
    kind, text = fetch_resource(url_or_path)
    if not text.strip():
        print(f"  could not read any content from {url_or_path}")
        return
    print(f"  read {kind} ({len(text)} chars)")
    item = {
        "title": os.path.basename(url_or_path),
        "snippet": text.strip()[:500],
        "link": url_or_path,
    }
    resource_text = f"[{kind}] {url_or_path}\n{text.strip()[:MAX_CONTENT_CHARS]}"
    try:
        resource = _extract(item, resource_text)
        _store(resource, item, resource_text)
    except (ValidationError, Exception) as e:
        print(f"  failed to ingest {url_or_path}: {e}")
    print("done.")
