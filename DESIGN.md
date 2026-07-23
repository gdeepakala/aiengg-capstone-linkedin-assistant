# LinkedIn Knowledge Assistant — Design Document

**Author:** Bharath D | **Cohort:** AIEngg 2026 | **Submitted:** July 2026

---

## 1. Problem Specificity

LinkedIn posts are ephemeral — they surface once in the feed and disappear. There is no semantic search on LinkedIn, and Google returns de-contextualized snippets. The specific pain point: a post about a relevant project, tool, or role is seen once and cannot be found again.

**Narrow problem statement:** Build a personal knowledge base that captures LinkedIn posts from two specific communities (AIEngg cohort, eBPF ecosystem) and makes them queryable by meaning — not by exact keyword.

This is not "an AI that helps with everything." It solves one problem: retrieval of community knowledge that is otherwise lost.

---

## 2. Data Processing — Ingestion Pipeline

```
User query (e.g. "site:linkedin.com AIEngg capstone")
  → Serper.dev (Google Search API) — returns top 10 results
  → For each result:
      → Extract URLs from title + snippet
      → Resolve shortlinks (lnkd.in → real destination)
      → Skip linkedin.com links (login-walled, yields boilerplate)
      → Fetch supported resources:
          GitHub   → README via raw.githubusercontent.com
          YouTube  → transcript via youtube-transcript-api
          arXiv    → abstract + metadata via arXiv API
          PDF      → full text via pypdf
          Web      → cleaned text via requests
      → LLM (gpt-5.4-mini) extracts structured metadata from snippet + resource
      → Pydantic validates schema — rejects malformed output
      → text-embedding-3-small embeds (author + topic + tags + content)
      → ChromaDB upserts document + embedding + metadata
```

**Direct resource ingestion:** Users can paste a URL (GitHub repo, YouTube video, arXiv paper, PDF, web page) directly. The LLM router detects the URL and bypasses the search step, ingesting the resource in full. Questions backed by directly ingested resources consistently score 5/5 vs 1–3/5 for snippet-only posts.

**Why Serper instead of scraping LinkedIn:** LinkedIn blocks all unauthenticated access. Serper uses Google's index of LinkedIn pages — the only reliable ingestion path without authentication.

---

## 3. Retrieval Design

```
User question
  → text-embedding-3-small embeds question (1536 dimensions)
  → ChromaDB cosine similarity search → top 3 documents
  → Build context: author + topic + document text + URL per result
  → gpt-5.4-mini generates grounded answer with citations
```

**Embedding strategy:** The stored embedding includes author name, topic, tags, and document content — not just document content alone. This ensures author-specific queries ("what did Lavanya build?") retrieve the right document even when the document text doesn't repeat the author's name prominently.

**Why ChromaDB:** Local, persistent, no infrastructure required. Suitable for a personal knowledge base where the corpus is hundreds to low thousands of documents.

---

## 4. Orchestration — RAG Pipeline with LLM Router

```
User input (any natural language or URL)
  → LLM router (gpt-5.4-mini) classifies intent: ingest or retrieve
      ingest + URL  → direct resource ingestion (GitHub/YouTube/arXiv/PDF/web)
      ingest + query → Serper search ingestion
      retrieve      → semantic search + LLM answering
```

**Why an LLM router:** A rule-based router cannot distinguish `"site:linkedin.com AIEngg capstone"` (ingest intent) from `"what did Lavanya build?"` (retrieve intent) — both are plain text. The LLM router also strips filler words from user input before passing the clean query to the search or retrieval step.

The three components behind the router:

**Ingestion** (`agents/ingestion.py`)
- Serper search → top 10 results
- For each result: follow GitHub/YouTube/arXiv/PDF links, fetch full content
- LLM extracts structured metadata → Pydantic validates → ChromaDB upserts

**Resource fetcher** (`agents/resources.py`)
- Dispatches by URL type: GitHub → raw README, YouTube → transcript, arXiv → abstract, PDF → pypdf, web → requests
- Resolves shortlinks before dispatch; skips linkedin.com (login-walled)

**Retrieval** (`agents/retrieval.py`)
- ChromaDB cosine similarity → top 3 documents
- LLM generates grounded answer with citations

No LangGraph or orchestration framework used — complexity was chosen to match the problem.

---

## 5. Evaluations

Three approaches compared across 10 ground-truth questions. Questions were written from raw LinkedIn post content before ingestion — not derived from system output — to ensure independence.

| Approach | Method | Score |
|---|---|---|
| Keyword Baseline | Keyword count across all docs, return top 3 | Precision@3: 0.90 |
| Vanilla RAG | ChromaDB semantic search, return raw docs | Precision@3: 0.90 |
| Full Pipeline | Semantic search + LLM answer generation | LLM-as-judge avg: 3.80/5 |

**LLM-as-judge rubric (1–5):**
- 5: Correct, specific, cites source
- 4: Correct, minor detail missing
- 3: Partially correct
- 2: Mostly wrong
- 1: Completely wrong

**Key finding:** Retrieval is not the bottleneck — P@3 ≈ 0.90 for both baseline and RAG. Answer quality tracks source depth: GitHub README / full web page → 5/5; targeted query with specific phrase → 5/5; generic Serper snippet → 1–3/5. The system's ceiling is content depth, not retrieval accuracy.

**Justified final choice:** The full pipeline is chosen over keyword and vanilla RAG because P@3 alone cannot distinguish answer quality — both simpler approaches retrieve the right documents but return raw text the user must interpret themselves. The LLM answer generation step synthesizes, cites sources, and handles the case where the answer is absent ("I couldn't find any posts about Rust") rather than returning irrelevant documents silently.

**LLM-as-judge limitation:** Judge model (gpt-5.4-mini) is the same family as the answering model. Per arXiv:2502.04313, same-family judges exhibit self-enhancement bias. A production eval would use a different model family as judge.

---

## 6. Decision Reasoning

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | ChromaDB | Local, no infra, persistent across sessions |
| Embeddings | text-embedding-3-small | Fast, cheap, 1536-dim, strong semantic quality |
| LLM | gpt-5.4-mini | Cohort API access; sufficient for extraction + answering |
| Search | Serper.dev | Only reliable LinkedIn ingestion path; 2,500 free searches |
| Validation | Pydantic | Structured output with type enforcement; rejects bad LLM output without crashing |
| Resource fetching | Custom (requests + pypdf + youtube-transcript-api) | No LangChain dependency; full control over what gets stored |
| Orchestration | LLM router (gpt-5.4-mini) | Rule-based routing can't distinguish ingest queries from retrieve questions — both are plain text. LLM router also cleans filler words before passing to search/retrieval. |
| Baseline | Keyword search | Required to show retrieval improvement; established before adding complexity |
| Observability | LangSmith | Traces every LLM call (router, extraction, answering) with latency and token counts — makes the pipeline observable and debuggable |

---

## Failure Analysis

| Failure | Root cause | Fix |
|---|---|---|
| Repost attribution | Gaurav Sen reposted Lavanya Mothilal's winning post. Google titles the result "Gaurav Sen's Post"; the snippet says "My capstone project won 1st Prize." The LLM sees the title name and stores `author=Gaurav Sen`. Added prompt instruction to detect first-person content that doesn't match the title name. Reliable fix: targeted query `site:linkedin.com "Lavanya Mothilal" "family financial tracker"` surfaces Lavanya's own post where title is "Lavanya Mothilal's Post." | Prompt-level reshare detection + targeted ingestion query |
| Cross-post synthesis | Answer spans multiple posts; top-1 retrieved | Increase n_results; merge on aggregation signals |
| Instructor attribution | Role detail in low-ranked post | Extract role fields as explicit metadata |
| Thin snippet answers | No linked repo in post | Auto-search per author for GitHub repo |

---

## v2 — Agent Ingestion Loop

After the v1 submission, the ingestion pipeline was extended to an **agentic loop**: after fetching each resource, the LLM inspects the content and decides which URLs are worth following next (GitHub repos, arXiv papers, YouTube videos). A BFS queue tracks depth — Serper results are depth 0, discovered links are depth 1+.

**v1 eval score: 3.80/5 → v2 eval score: 3.50/5**

The loop did not improve scores. Understanding why is more instructive than the numbers.

### The real ceiling: LinkedIn via Google Search

The ingestion pipeline uses Serper (Google Search API) because LinkedIn blocks all unauthenticated access. This introduces a structural depth limit that no amount of pipeline sophistication can overcome:

| What you want | What Serper gives you |
|---|---|
| Full post body (~500–2000 words) | 150–300 char Google snippet |
| lnkd.in shortlinks in post body | Invisible — snippet truncated before the link |
| YouTube videos embedded in post | Invisible — only visible inside the post |
| Architecture details in post | Invisible — behind the login wall |

Every interesting piece of content in a LinkedIn post — the links, the code snippets, the architecture descriptions — lives in the post body. Google indexes the post but only the snippet is public. The agent loop can only follow URLs it can see. It cannot see inside LinkedIn posts.

### Why v2 hurt scores

The agent loop re-fetched GitHub repos at depth 1 (discovered inside Serper snippets) and stored them a second time — but with degraded metadata. At depth 0, a Serper result includes title and snippet; the LLM extracts `author=Tamilarasu Ravi`, `topic=ReconAI`. At depth 1, the system only has the raw URL; the LLM stores `author=Unknown`, `topic=Financial Operations Platform`. ChromaDB now holds two copies: one with good metadata and one with bad. The bad copy ranks higher on some queries because it contains more raw text — pulling retrieval in the wrong direction.

### What would actually help

| Failure | Fix | Category |
|---|---|---|
| LinkedIn post body inaccessible | Proxycurl (paid) or manual copy-paste at capture time | Data access |
| lnkd.in links invisible | Paste the shortlink directly — system resolves it | Data access |
| Long documents (50k README) embedded as one vector | Chunk into 500-char pieces, embed each | Advanced RAG |
| Exact names (pgvector, Prabrisha) missed by semantic search | Hybrid search: BM25 + semantic | Advanced RAG |
| Judge list split across two posts | Increase n_results; merge on aggregation signals | Retrieval design |

The data access problems cannot be solved by better retrieval. The RAG improvements (chunking, hybrid search) would raise scores on questions already backed by ingested content — but the majority of failures in this eval are data access failures, not retrieval failures.

---

## Getting API Keys

**OpenAI:** platform.openai.com → API Keys → Create new secret key → `OPENAI_API_KEY=sk-...`

**Serper:** serper.dev → Sign up (2,500 free searches) → Dashboard → API Key → `SERPER_API_KEY=...`
