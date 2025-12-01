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
    """Find exact PR/repo/ticket/commit/markdown matches BEFORE FAISS."""
    
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

        # ---- Body text ----
        if email.body:
            if normalize(query) in normalize(email.body):
                matches.append(i)
                continue

        # ---- Markdown sections ----
        md = getattr(email, "markdown", None)
        if md:

            # ---- Headings ----
            if md.get("headings"):
                if any(normalize(h["title"]) in q for h in md["headings"]):
                    matches.append(i)
                    continue

            # ---- Code blocks ----
            if md.get("code_blocks"):
                if any(q in normalize(cb) for cb in md["code_blocks"]):
                    matches.append(i)
                    continue

            # ---- Bullet lists ----
            if md.get("lists"):
                if any(q in normalize(li) for li in md["lists"]):
                    matches.append(i)
                    continue

    return matches


def semantic_search(query: str, k: int = TOP_K) -> List[int]:
    """FAISS semantic search fallback."""
    # small boost so that code-block queries work better
    query = query + " markdown code block heading list"
    query_emb = MODEL.encode([query])
    distances, indices = INDEX.search(query_emb, k)
    return indices[0].tolist()


# ============================================================
#       FORMAT RETRIEVED EMAILS INTO LLM CONTEXT CHUNKS
# ============================================================

def get_email_chunks(indices: List[int]) -> List[str]:
    chunks = []

    for idx in indices:
        email = META[idx]
        md = getattr(email, "markdown", None)

        # ---- Markdown formatted section ----
        md_section = ""
        if md:
            if md.get("headings"):
                heads = "\n".join(
                    [f"  - {'#'*h['level']} {h['title']}" for h in md["headings"]]
                )
                md_section += f"\nMarkdown Headings:\n{heads}\n"

            if md.get("code_blocks"):
                codes = "\n\n".join(
                    [f"--- code block {i+1} ---\n{cb[:400]}"   # limit size
                     for i, cb in enumerate(md["code_blocks"])]
                )
                md_section += f"\nMarkdown Code Blocks:\n{codes}\n"

            if md.get("lists"):
                lists = "\n".join([f"  - {li}" for li in md["lists"]])
                md_section += f"\nMarkdown Lists:\n{lists}\n"

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

{md_section}

Body:
{email.body[:2000]}
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

Here are the retrieved email fragments from GitHub notification emails.
Markdown headings, code blocks, and lists may be included where present.

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