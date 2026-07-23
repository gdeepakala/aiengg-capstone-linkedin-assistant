import json
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langsmith.wrappers import wrap_openai
from agents.ingestion import ingest, ingest_resource
from agents.retrieval import retrieve

client = wrap_openai(OpenAI())


def route_intent(user_input):
    response = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": """You are a router for a LinkedIn knowledge base system.
Classify the user's intent as 'ingest' or 'retrieve', and extract a clean query.

ingest: user wants to find or save posts/content
  examples: "site:linkedin.com AIEngg capstone", "save eBPF posts by Liz Rice", "https://github.com/owner/repo"

retrieve: user wants to ask a question and get an answer
  examples: "what did Lavanya build?", "who won the hackathon?", "what is eBPF used for?"

Rules for extracting the query:
- Remove filler words: "find and ingest", "please", "tell me about", "ingest only this", "save this", "add this"
- If the input contains a URL, return the URL exactly as the query
- For ingest: return a clean search query (keep site:linkedin.com if present)
- For retrieve: return the question as cleanly as possible

Return JSON only: {"intent": "ingest" or "retrieve", "query": "<clean query or URL>"}""",
            },
            {"role": "user", "content": user_input},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    print("LinkedIn Knowledge Base — v1 (single-pass ingestion)")
    print("Ask a question or say what to save. Keep it simple.\n")
    print("  site:linkedin.com AIEngg capstone Lavanya    — save posts")
    print("  https://github.com/owner/repo                — save a resource")
    print("  what did Lavanya build?                      — ask a question")
    print("  who won the India AI Hackathon 2026?         — ask a question")
    print("\nType 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("quit", "q", "exit"):
            break

        result = route_intent(user_input)
        intent = result.get("intent")
        query = result.get("query", user_input)

        if intent == "ingest":
            if query.startswith("http://") or query.startswith("https://"):
                print(f"\n[routing to direct resource ingestion: {query}]")
                ingest_resource(query)
            else:
                print(f"\n[routing to search ingestion: {query}]")
                ingest(query, max_depth=0)
        elif intent == "retrieve":
            print(f"\n[routing to retrieval]")
            answer = retrieve(query)
            print("\n" + answer + "\n")
        else:
            print("Could not determine intent — please rephrase.")


if __name__ == "__main__":
    main()
