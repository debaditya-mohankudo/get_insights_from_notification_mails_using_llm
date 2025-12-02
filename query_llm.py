import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional, Tuple
import ollama
import re

from email_models import EmailMessage


# ============================================================
#                    LOAD INDEX + METADATA
# ============================================================

with open("meta.pkl", "rb") as f:
    META: List[EmailMessage] = pickle.load(f)

index = faiss.read_index("index.faiss")
model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
#                 QUERY CLASSIFICATION HELPERS
# ============================================================

COMMIT_REGEX = r"\b[a-f0-9]{7,40}\b"


def extract_commit_hash(query: str) -> Optional[str]:
    """Extract commit hash-like tokens."""
    q = query.lower()
    m = re.search(COMMIT_REGEX, q)
    if m:
        return m.group(0)
    return None


def extract_pr_number(query: str) -> Optional[int]:
    """
    Extract PR number *only* when the query explicitly mentions PR.
    Prevent false activation when query contains commit-like tokens.
    """

    q = query.lower()

    # If query contains any commit hash, do NOT activate PR mode.
    if re.search(COMMIT_REGEX, q):
        return None

    # Explicit PR-only patterns
    pr_patterns = [
        r"pr\s*#\s*(\d+)",
        r"pull\s*request\s*#\s*(\d+)",
        r"pull\s*#\s*(\d+)",
        r"\bpr\s+(\d+)\b",
    ]

    for pat in pr_patterns:
        m = re.search(pat, q)
        if m:
            return int(m.group(1))

    return None


# ============================================================
#                 EXACT-MATCH RANKING
# ============================================================

def score_email(query: str, email: EmailMessage) -> float:
    """Weighted exact match scoring."""
    q = query.lower()
    score = 0.0

    # PR numbers
    if email.pr_numbers:
        for pr in email.pr_numbers:
            if str(pr) in q:
                score += 6

    # Repo names
    if email.repos:
        for r in email.repos:
            if r.lower() in q:
                score += 3

    # Tickets
    if email.tickets:
        for t in email.tickets:
            if t.lower() in q:
                score += 2

    # Commits (7-char abbreviated)
    if email.commits:
        for c in email.commits:
            if c.lower() in q:
                score += 4

    # File paths
    if email.files_modified:
        for path in email.files_modified:
            if path.lower() in q:
                score += 4

    # PR title
    if email.pr_title and email.pr_title.lower() in q:
        score += 5

    return score


# ============================================================
#                 QUERY → VECTOR SEARCH
# ============================================================

def search_semantic(query: str, top_k: int = 10) -> List[int]:
    vec = model.encode([query])
    _, idx = index.search(vec, top_k)
    return idx[0].tolist()


# ============================================================
#             FORMAT RESULTS → CONTEXT FOR LLM
# ============================================================

def build_context(emails: List[EmailMessage]) -> str:
    parts = []

    for e in emails:
        block = []

        if e.pr_numbers:
            block.append(f"PR Numbers: {e.pr_numbers}")

        if e.repos:
            block.append(f"Repos: {e.repos}")

        if e.tickets:
            block.append(f"Tickets: {e.tickets}")

        if e.pr_title:
            block.append(f"Title: {e.pr_title}")

        if e.commits:
            block.append("Commits:\n" + "\n".join(f"- {c}" for c in e.commits))

        if e.files_modified:
            block.append("Files Changed:\n" + "\n".join(f"- {c}" for c in e.files_modified))

        # include markdown extracted content
        if e.markdown:
            block.append("Markdown Sections:\n" + "\n".join(e.markdown))

        # trimmed email body
        body_preview = e.body[:2000]
        block.append("Email Body:\n" + body_preview)

        parts.append("\n\n".join(block))

    return "\n\n============================\n\n".join(parts)


# ============================================================
#                        MAIN LOGIC
# ============================================================

def answer_query(query: str):
    query = query.strip()

    # 1️⃣ Detect commit-only mode
    commit = extract_commit_hash(query)
    if commit:
        print(f"[Commit mode → commit {commit}]")

        # Retrieve all emails referencing this commit
        matched = [e for e in META if e.commits and any(commit in c for c in e.commits)]

        if not matched:
            print("No emails found for this commit.")
            return

        context = build_context(matched)
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"You are an expert summarizer.\nSummarize details about commit {commit} using the following context only.\n\n{context}"
        )

        print(response["response"])
        return

    # 2️⃣ Detect PR mode safely
    pr = extract_pr_number(query)
    if pr:
        print(f"[PR mode activated → PR #{pr}]")

        # Filter emails strictly by PR
        emails = [e for e in META if e.pr_numbers and pr in e.pr_numbers]

        if not emails:
            print("No emails found for this PR.")
            return

        context = build_context(emails)

        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"You are an expert PR analyst.\nSummarize PR #{pr} using ONLY the context below.\n\n{context}"
        )
        print(response["response"])
        return

    # 3️⃣ Normal semantic search mode
    print("[Semantic mode → no PR/commit detected]")

    emb = model.encode([query])
    _, idx = index.search(emb, 5)
    selected = [META[i] for i in idx[0]]

    context = build_context(selected)

    response = ollama.generate(
        model="llama3.2:3b",
        prompt=f"You are an expert summarizer.\nAnswer using ONLY the context below.\n\n{context}"
    )

    print(response["response"])


# ============================================================
#                COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please pass a query string.")
        sys.exit(1)

    q = " ".join(sys.argv[1:])
    answer_query(q)