from dotenv import load_dotenv
import os

load_dotenv()

from agents.ingestion import ingest, ingest_resource
from agents.retrieval import retrieve

def main():
    print("LinkedIn Knowledge base");
    print("1. Ingest posts");
    print("2. Search posts");
    print("3. Ingest a resource (URL or PDF path)");
    choice = input("Choose (1, 2 or 3): ").strip()

    if choice=="1":
        query=input("Enter search query to find posts (eg. 'site:linkedin.com AIengg capstone'):").strip()
        ingest(query)
    elif choice=="2":
        question=input("Ask a question").strip()
        answer=retrieve(question)
        print("\n"+answer)
    elif choice=="3":
        src=input("Paste a URL (GitHub/YouTube/arXiv/PDF/web) or a local PDF path: ").strip()
        ingest_resource(src)
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
