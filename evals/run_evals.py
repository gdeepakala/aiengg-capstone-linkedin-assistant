import json
import os
import sys
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

client = OpenAI()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection("linkedin_posts")


def load_questions():
    with open("data/eval_questions.json") as f:
        return json.load(f)


# --- Tier 1: Keyword baseline ---
def keyword_search(question, n=3):
    all_docs = collection.get(include=["documents", "metadatas"])
    keywords = [w for w in question.lower().split() if len(w) > 3]
    scores = []
    for i, doc in enumerate(all_docs["documents"]):
        score = sum(1 for k in keywords if k in doc.lower())
        scores.append((score, i))
    scores.sort(reverse=True)
    top = scores[:n]
    return [all_docs["documents"][i] for _, i in top], [all_docs["metadatas"][i] for _, i in top]


# --- Tier 2: Vanilla RAG (embedding search, no LLM) ---
def vanilla_rag(question, n=3):
    emb = client.embeddings.create(model="text-embedding-3-small", input=question)
    results = collection.query(
        query_embeddings=[emb.data[0].embedding],
        n_results=n,
        include=["documents", "metadatas"],
    )
    return results["documents"][0], results["metadatas"][0]


# --- Tier 3: Full pipeline (RAG + LLM answer) ---
def full_pipeline(question):
    from agents.retrieval import retrieve
    return retrieve(question)


# --- Precision@3: expected keywords found in any retrieved doc ---
def precision_at_3(docs, expected):
    keywords = [w for w in expected.lower().split() if len(w) > 3]
    for doc in docs:
        if any(k in doc.lower() for k in keywords):
            return 1
    return 0


# --- LLM-as-judge ---
def llm_judge(question, answer, expected):
    prompt = f"""Score this answer 1-5 against the expected answer.

Question: {question}
Expected: {expected}
Answer: {answer}

Rubric:
5 - Correct, specific, cites source
4 - Correct, missing minor detail
3 - Partially correct
2 - Mostly wrong
1 - Completely wrong or no answer

Return JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""

    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


def run_evals():
    questions = load_questions()
    results = []

    for q in questions:
        question = q["question"]
        expected = q["expected"]
        print(f"\nQ: {question}")

        kw_docs, _ = keyword_search(question)
        kw_p3 = precision_at_3(kw_docs, expected)

        rag_docs, _ = vanilla_rag(question)
        rag_p3 = precision_at_3(rag_docs, expected)

        answer = full_pipeline(question)
        judge = llm_judge(question, answer, expected)

        results.append({
            "question": question,
            "keyword_p3": kw_p3,
            "rag_p3": rag_p3,
            "full_score": judge["score"],
            "reason": judge["reason"],
        })
        print(f"  Keyword P@3: {kw_p3} | RAG P@3: {rag_p3} | LLM score: {judge['score']} — {judge['reason']}")

    print("\n=== EVAL SUMMARY ===")
    n = len(results)
    print(f"Keyword Baseline — Precision@3: {sum(r['keyword_p3'] for r in results)/n:.2f}")
    print(f"Vanilla RAG      — Precision@3: {sum(r['rag_p3'] for r in results)/n:.2f}")
    print(f"Full Pipeline    — LLM-as-judge avg: {sum(r['full_score'] for r in results)/n:.2f}/5")


if __name__ == "__main__":
    run_evals()
