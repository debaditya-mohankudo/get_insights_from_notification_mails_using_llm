from __future__ import annotations

import sys
import pickle
from dataclasses import dataclass
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama


LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "all-MiniLM-L6-v2"


# ------------------------------------------------------
# SAME DATACLASS FROM build_index.py
# ------------------------------------------------------
@dataclass
class EmailMessage:
    subject: str
    sender: str
    date: str
    body: str
    commits: List[str] | None = None
    pr_numbers: List[str] | None = None
    repos: List[str] | None = None
    issues: List[str] | None = None
    sql_queries: List[str] | None = None
    files_modified: List[str] | None = None   # ← NEW FIELD
    tags: List[str] | None = None

    def full_text(self) -> str:
        # This is unused here, as embeddings already exist
        return f"{self.subject}\n\n{self.body}"


# ------------------------------------------------------
# LOAD INDEX + METADATA
# ------------------------------------------------------
def load_metadata(path: str = "meta.pkl") -> List[EmailMessage]:
    with open(path, "rb") as f:
        messages = pickle.load(f)
    return messages


def load_index(path: str = "index.faiss") -> faiss.Index:
    return faiss.read_index(path)


# ------------------------------------------------------
# FAISS SEMANTIC SEARCH
# ------------------------------------------------------
def semantic_search(
    query: str,
    messages: List[EmailMessage],
    index: faiss.Index,
    top_k: int = 5,
) -> List[EmailMessage]:

    encoder = SentenceTransformer(EMBED_MODEL)
    q_vec: np.ndarray = encoder.encode([query]).astype("float32")

    distances, indices = index.search(q_vec, top_k)
    return [messages[i] for i in indices[0]]


# ------------------------------------------------------
# BUILD RAG CONTEXT (NOW SHOWS ALL STRUCTURED DATA)
# ------------------------------------------------------
def build_context(results: List[EmailMessage]) -> str:
    ctx = ""

    for i, r in enumerate(results):

        ctx += f"\n--- EMAIL {i+1} ---\n"
        ctx += f"Subject: {r.subject}\n"
        ctx += f"From: {r.sender}\n"
        ctx += f"Date: {r.date}\n"
        ctx += f"Tags: {', '.join(r.tags) if r.tags else 'None'}\n\n"

        # Body
        ctx += r.body[:5000] + "\n\n"

        # Commits
        if r.commits:
            ctx += "Commits:\n"
            for c in r.commits:
                ctx += f"  - {c}\n"
            ctx += "\n"

        # PR Numbers
        if r.pr_numbers:
            ctx += "PRs: " + ", ".join(r.pr_numbers) + "\n\n"

        # Repos
        if r.repos:
            ctx += "Repos: " + ", ".join(r.repos) + "\n\n"

        # Issues
        if r.issues:
            ctx += "Issues: " + ", ".join(r.issues) + "\n\n"

        # SQL Queries
        if r.sql_queries:
            ctx += "SQL Queries:\n"
            for sql in r.sql_queries:
                ctx += f"{sql}\n\n"

        # Files Modified
        if r.files_modified:
            ctx += "Files Modified:\n"
            for f in r.files_modified:
                ctx += f"  - {f}\n"
            ctx += "\n"
    return ctx


# ------------------------------------------------------
# ORIGINAL ask_llm() (UNCHANGED)
# ------------------------------------------------------
def ask_llm(query, context):
    prompt = f"""
You are an assistant reading github notification emails from PRs.

User query:
{query}

Relevant emails:
{context}

Answer the user's question concisely by analyzing these emails.
Extract important details such as:
- what the email is about
- links
- actions requested (PR merged, review requested, security issue, bug fix, performance improvement, etc.)
- summary of conversation if multiple emails
- final actionable insights

Return a clean explanation.
"""

    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_llm.py \"search text\" [k]")
        sys.exit(1)

    query: str = sys.argv[1]
    k: int = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    messages = load_metadata()
    index = load_index()

    results = semantic_search(query, messages, index, top_k=k)
    context = build_context(results)

    answer = ask_llm(query, context)

    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50 + "\n")

