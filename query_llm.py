import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
import ollama

from email_models import EmailMessage


# ============================================================
#                    LOAD INDEX + METADATA
# ============================================================

with open("meta.pkl", "rb") as f:
    META = pickle.load(f)

INDEX = faiss.read_index("index.faiss")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TOP_K = 5   # semantic fallback


# ============================================================
#                    HELPER FUNCTIONS
# ============================================================

def normalize(text: str) -> str:
    """Make matching easier."""
    return text.lower().replace("#", "").strip()


def find_exact_matches(query: str) -> List[int]:
    """Find exact PR/repo/ticket/commit matches from metadata BEFORE FAISS."""
    
    q = normalize(query)
    matches = []

    for i, email in enumerate(META):

        # ---- PR numbers ----
        if email.pr_numbers:
            if any(normalize(p) in q or q in normalize(p) for p in email.pr_numbers):
                matches.append(i)
                continue

        # ---- Tickets ----
        if email.tickets:
            if any(normalize(t) in q for t in email.tickets):
                matches.append(i)
                continue

        # ---- Repo names ----
        if email.repos:
            if any(normalize(r) in q for r in email.repos):
                matches.append(i)
                continue

        # ---- Commit SHAs ----
        if email.commits:
            if any(normalize(c.split()[0]) in q for c in email.commits):
                matches.append(i)
                continue

        # ---- File paths ----
        if email.files_modified:
            if any(normalize(f) in q for f in email.files_modified):
                matches.append(i)
                continue

        # ---- PR title ----
        if email.pr_title:
            if normalize(email.pr_title) in q:
                matches.append(i)
                continue
        
        if email.body:
            if normalize(email.body) in q:
                matches.append(i)
                continue

    return matches


def semantic_search(query: str, k: int = TOP_K) -> List[int]:
    """FAISS semantic search fallback."""
    query_emb = MODEL.encode([query])
    distances, indices = INDEX.search(query_emb, k)
    return indices[0].tolist()


def get_email_chunks(indices: List[int]) -> List[str]:
    """Return formatted text for LLM."""
    chunks = []

    for idx in indices:
        email = META[idx]

        block = f"""
Subject: {email.subject}
From: {email.sender}
Date: {email.date}

PR Numbers: {email.pr_numbers}
Repo: {email.repos}
Tickets: {email.tickets}
PR Title: {email.pr_title}

Commits:
{email.commits}

Files Modified:
{email.files_modified}

Change Counts:
{email.change_counts}

Body:
{email.body[:2000]}  # limit for safety
"""
        chunks.append(block)

    return chunks


# ============================================================
#                       ASK LLM
# ============================================================

def ask_llm(query: str, chunks: List[str]) -> str:
    """Send the retrieved chunks + user query to local LLM via Ollama."""

    prompt = f"""
You are an expert GitHub PR analyst.

Here are the retrieved email fragments from GitHub notification emails:

{'-'*60}
{chr(10).join(chunks)}
{'-'*60}

Based on ONLY this information, answer the user query:

User query: "{query}"

Produce a structured, accurate, concise answer.
If PR numbers or commits are missing in data, state that clearly.
    """

    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )

    return response["response"]


# ============================================================
#                       MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_llm.py \"your question\"")
        sys.exit()

    query = sys.argv[1]

    # ---- Step 1: exact match based on metadata ----
    exact = find_exact_matches(query)

    if exact:
        print(f"[Exact-match retrieval → {len(exact)} results]")
        indices = exact[:TOP_K]
    else:
        # ---- Step 2: semantic search fallback ----
        print("[Semantic retrieval]")
        indices = semantic_search(query)

    # ---- Step 3: collect email blocks ----
    chunks = get_email_chunks(indices)

    # ---- Step 4: ask LLM ----
    answer = ask_llm(query, chunks)

    print("\n==================================================")
    print(answer)
    print("==================================================\n")