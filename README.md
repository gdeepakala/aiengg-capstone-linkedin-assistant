# LinkedIn Knowledge Assistant

A multi-agent system that ingests LinkedIn posts from communities you follow into a vector database and lets you query them by meaning — not by keyword.

**Problem:** LinkedIn posts surface once in your feed and disappear. There is no semantic search on LinkedIn. Google returns noise. This system captures posts at ingestion time and makes them queryable forever.

**Communities indexed:**
- AIEngg cohort (Gaurav Sen, Tanishq Singh) — capstone projects, architectures, hackathons
- eBPF ecosystem (Liz Rice, Isovalent) — beginner resources, tools, papers

---

## Architecture

```
Ingestion
  User query (e.g. "site:linkedin.com AIEngg capstone")
      → Serper.dev search → top 10 results
      → For each result: follow GitHub/YouTube/arXiv/PDF links
      → LLM (gpt-5.4-mini) extracts metadata → Pydantic validates
      → text-embedding-3-small embeds content
      → ChromaDB stores embedding + metadata + full text

Retrieval
  User question
      → text-embedding-3-small embeds question
      → ChromaDB semantic search → top 3 documents
      → LLM (gpt-5.4-mini) generates grounded answer with citations
```

### Files
```
agents/
  ingestion.py    — Serper search, LLM extraction, ChromaDB upsert
  retrieval.py    — semantic search + LLM answering
  resources.py    — GitHub README / YouTube / arXiv / PDF / web fetcher
data/
  eval_questions.json   — 10 ground-truth Q&A pairs from raw LinkedIn posts
evals/
  run_evals.py    — three-tier comparison: keyword vs RAG vs full pipeline
main.py           — CLI menu (ingest / search / ingest resource)
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:
```
OPENAI_API_KEY=...
SERPER_API_KEY=...
```
## Getting API Keys

**OpenAI API Key**
1. Go to platform.openai.com → API Keys → Create new secret key
2. Add to `.env` as `OPENAI_API_KEY=sk-...`

**Serper API Key**
1. Go to serper.dev → Sign up (free tier: 2,500 searches)
2. Dashboard → API Key → copy
3. Add to `.env` as `SERPER_API_KEY=...`

Your `.env` file should look like:
```
OPENAI_API_KEY=sk-...
SERPER_API_KEY=...
```

Run:
```bash
python main.py
```

---

## Evaluation

Three approaches compared across 10 ground-truth questions written from raw LinkedIn post content (not from system output).

| Approach | Metric | Score |
|---|---|---|
| Keyword Baseline | Precision@3 | 1.00 |
| Vanilla RAG | Precision@3 | 1.00 |
| Full Pipeline | LLM-as-judge (1–5) | 3.10 / 5 |

**Why P@3 = 1.00 for both baseline and RAG:** The relevant documents are retrieved correctly by both approaches — the bottleneck is not retrieval but answer quality. Keyword search and semantic search both surface the right documents. The full pipeline's LLM-as-judge score reveals where quality falls short.

**Per-question breakdown:**

| Question | Score | Note |
|---|---|---|
| What architecture does ReconAI use? | 5/5 | GitHub README ingested |
| What database does ReconAI use? | 5/5 | GitHub README ingested |
| What is eBPF used for? | 5/5 | ebpf.io full page ingested |
| Who won the India AI Hackathon 2026? | 4/5 | Correct winner, missing specifics |
| Who won 1st prize in AIEngg capstone? | 3/5 | Correct but hedged |
| What did Lavanya Mothilal build? | 2/5 | Snippet only, no repo link |
| Who judged the India AI Hackathon 2026? | 2/5 | Cross-post synthesis failure |
| Who teaches the AIEngg cohort? | 2/5 | Instructor attribution failure |
| How long is the AIEngg cohort? | 2/5 | Conflicting duration in posts |
| What did Prabrisha Chattopadhyay build? | 1/5 | Post not in top 3 |

**Key finding:** Answer quality tracks source depth. Posts backed by a GitHub README or full web page score 5/5. Posts ingested from search snippets alone score 1–2/5. The system's ceiling is the quality of ingested content.

Run evals:
```bash
python evals/run_evals.py
```

**Why some questions score 5/5:** The highest-scoring answers came from directly ingested resources — the ReconAI GitHub README (~50K chars, fetched automatically when its link appeared in search results) and the ebpf.io what-is-ebpf page (22K chars, ingested via option 3). Questions backed by these rich documents scored 5/5. Questions relying only on Serper search snippets (150–300 chars) scored 1–2/5.

**Direct resource ingestion (option 3)** accepts GitHub repos, YouTube videos, arXiv papers, PDFs, and web pages — and is the highest-leverage way to improve answer quality for a specific topic.

---

## Failure Analysis

### 1. Repost attribution bug
Gaurav Sen reposted Lavanya Mothilal's winning post. Google titles the result "Gaurav Sen's Post" and the snippet says "My capstone project won 1st Prize." The LLM stores `author=Gaurav Sen` instead of Lavanya Mothilal.

**Fix:** Detect reshare/repost patterns in the snippet before extracting author. Phrases like "reposted this" or "shared this" signal the author is not the original creator.

### 2. Cross-post synthesis failure
The full judge list for the India AI Hackathon 2026 is split across two posts — Tanishq Singh's judging post and Ambar Kashyap's winner post. The retrieval system returns the most semantically similar document, missing the other. Answer: "Tanishq Singh" instead of all 7 judges.

**Fix:** Increase `n_results` from 3 to 5, or add post-retrieval merging when the question contains aggregation signals ("who all", "list of", "full list").

### 3. Instructor attribution failure
General AIEngg cohort posts consistently mention Gaurav Sen by name. The specific registration post listing "Instructor: Tanishq Singh" ranks lower semantically. The system answers "Gaurav Sen teaches" when Tanishq Singh is the listed instructor.

**Fix:** During ingestion, extract structured role fields (instructor, founder, TA) explicitly and store them as separate metadata fields for direct lookup.

### 4. Thin snippet answers
LinkedIn is login-walled. Serper snippets are ~150–300 characters. Posts without a linked GitHub repo or paper yield surface-level answers. Dwaipayan Gupta's certificate post had no project link — zero technical detail available.

**Fix:** Auto-run a secondary search per author ("`NAME AIEngg capstone github`") during ingestion to find their project repo and ingest the README.

---

## Concepts demonstrated

- RAG (ChromaDB semantic search as core retrieval)
- Resource fetching (GitHub README, YouTube transcripts, arXiv, PDF, web pages)
- Structured output + validation (Pydantic)
- LLM-as-judge evaluation
- Three-tier eval comparison (keyword baseline → vanilla RAG → full pipeline)
- Multi-agent ingestion pipeline (search → extract → validate → embed → store)
