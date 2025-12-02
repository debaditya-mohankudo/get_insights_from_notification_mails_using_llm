import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional
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
    q = query.lower()
    m = re.search(COMMIT_REGEX, q)
    return m.group(0) if m else None


def extract_pr_number(query: str) -> Optional[int]:
    """
    Extract PR number only when explicitly mentioned.
    Avoid false activation when query contains commit-like hashes.
    """
    q = query.lower()

    if re.search(COMMIT_REGEX, q):
        return None

    patterns = [
        r"pr\s*#\s*(\d+)",
        r"pull\s*request\s*#\s*(\d+)",
        r"pull\s*#\s*(\d+)",
        r"\bpr\s+(\d+)\b",
    ]

    for pat in patterns:
        m = re.search(pat, q)
        if m:
            return int(m.group(1))

    return None


# ============================================================
#                 EXACT-MATCH RANKING
# ============================================================

def score_email(query: str, email: EmailMessage) -> float:
    q = query.lower()
    score = 0.0

    if email.pr_numbers:
        for pr in email.pr_numbers:
            if str(pr) in q:
                score += 6

    if email.repos:
        for r in email.repos:
            if r.lower() in q:
                score += 3

    if email.tickets:
        for t in email.tickets:
            if t.lower() in q:
                score += 2

    if email.commits:
        for c in email.commits:
            if c.lower() in q:
                score += 4

    if email.files_modified:
        for path in email.files_modified:
            if path.lower() in q:
                score += 4

    if email.pr_title and email.pr_title.lower() in q:
        score += 5

    if email.tags:
        for tag in email.tags:
            if tag.lower() in q:
                score += 2

    return score


# ============================================================
#                 SEMANTIC SEARCH + TAG RERANK
# ============================================================

def rerank_by_tags(query: str, emails: List[EmailMessage]) -> List[EmailMessage]:
    """
    Boost emails that have tags matching words in the query.
    Only affects semantic mode (NOT PR mode, NOT commit mode).
    """
    q = query.lower()

    for e in emails:
        boost = 0
        if e.tags:
            for t in e.tags:
                if t.lower() in q:
                    boost += 3  # Adjust this weight as needed
        e._tag_boost = boost

    return sorted(emails, key=lambda x: x._tag_boost, reverse=True)


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

        if e.tags:
            block.append(f"Tags: {', '.join(e.tags)}")

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

        if e.markdown:
            md_parts = []
            for section, items in e.markdown.items():
                if items:
                    md_parts.append(f"## {section}")
                    md_parts.extend(f"- {i}" for i in items)
            if md_parts:
                block.append("Markdown Sections:\n" + "\n".join(md_parts))

        block.append("Email Body:\n" + e.body)

        parts.append("\n\n".join(block))

    return "\n\n============================\n\n".join(parts)


# ============================================================
#                        MAIN LOGIC
# ============================================================

def answer_query(query: str):
    query = query.strip()

    # 1️⃣ Commit mode
    commit = extract_commit_hash(query)
    if commit:
        print(f"[Commit mode → commit {commit}]")

        matched = [e for e in META if e.commits and any(commit in c for c in e.commits)]
        if not matched:
            print("No emails found for this commit.")
            return

        context = build_context(matched)
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"You are an expert summarizer.\nSummarize details about commit {commit} using ONLY the context below.\n\n{context}"
        )
        print(response["response"])
        return

    # 2️⃣ PR mode
    pr = extract_pr_number(query)
    if pr:
        print(f"[PR mode activated → PR #{pr}]")

        emails = [e for e in META if e.pr_numbers and pr in e.pr_numbers]
        if not emails:
            print("No emails found for this PR.")
            return

        # NO tag reranking here — you requested PR mode should NOT be modified
        context = build_context(emails)
        response = ollama.generate(
            model="llama3.2:3b",
            prompt=f"You are an expert PR analyst.\nSummarize PR #{pr} using ONLY the context below.\n\n{context}"
        )
        print(response["response"])
        return

    # 3️⃣ Semantic mode
    print("[Semantic mode → no PR/commit detected]")

    vec = model.encode([query])
    _, idx = index.search(vec, 5)
    selected = [META[i] for i in idx[0]]

    # 🔥 Apply tag reranking ONLY in semantic search
    selected = rerank_by_tags(query, selected)

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