import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import ollama
import re

from _internal.email_models import EmailMessage
from _internal.tag_classifier import extract_tags_miniLM
from _internal.tags_from_file import classify_tags_from_files


# ============================================================
#                    LOAD INDEX + METADATA
# ============================================================
INDEX_DIR = "index_data"
with open(f"{INDEX_DIR}/meta.pkl", "rb") as f:
    META: List[EmailMessage] = pickle.load(f)

index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
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
            if c.message in q:
                score += 1

    if email.files_modified:
        for path in email.files_modified:
            if path.lower() in q:
                score += 3

    if email.pr_title and email.pr_title.lower() in q:
        score += 5

    if email.tags:
        for tag in email.tags:
            if tag.lower() in q:
                score += 2

    if email.contributors:
        for contributor in email.contributors:
            if contributor.lower() in q:
                score += 1

    return score


# ============================================================
#                 SEMANTIC SEARCH + TAG RERANK
# ============================================================

def rerank_by_tags(query: str, emails: List[EmailMessage]) -> List[EmailMessage]:
    """
    Boost emails that have tags matching words in the query.
    This version does NOT modify EmailMessage objects.
    It computes boost scores externally and sorts by them.
    """
    q = query.lower()

    def compute_boost(email: EmailMessage) -> int:
        if not email.tags:
            return 0

        boost = 0
        for t in email.tags:
            if t.lower() in q:
                boost += 3
        return boost

    # Sort using computed boost (highest first)
    return sorted(
        emails,
        key=lambda e: compute_boost(e),
        reverse=True
    )

def search_semantic(query: str, top_k: int = 10) -> List[int]:
    vec = model.encode([query])
    _, idx = index.search(vec, top_k)
    return idx[0].tolist()

def rerank_full(query, emails) -> List[EmailMessage]:
    return sorted(
        emails,
        key=lambda e: score_email(query, e),
        reverse=True
    )


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
            block.append("Commits:\n" + "\n".join(f"- {c.short} - {c.message}" for c in e.commits))

        if e.files_modified:
            block.append("Files Changed:\n" + "\n".join(f"- {c}" for c in e.files_modified))

        if e.markdown:
            md_parts = []
            for section, content in e.markdown.items():
                if content:
                    md_parts.append(f"## {section}")
                    md_parts.extend(f"- {content}")
            if md_parts:
                block.append("Markdown Sections:\n" + "\n".join(md_parts))

        if e.contributors:
            block.append(f"Contributors: {', '.join(e.contributors)}")

        block.append("Email Body:\n" + e.body)

        parts.append("\n\n".join(block))

    return "\n\n============================\n\n".join(parts)


# ============================================================
#                        MAIN LOGIC
# ============================================================

def answer_query(query: str):
    query = query.strip()
    tags_from_query = extract_tags_miniLM(query)
    tags_from_query_additional = classify_tags_from_files(re.findall(r'\b\w+\.\w{2,4}\b', query))
    tags_from_query.extend(tags_from_query_additional)
    print(f"Extracted tags from query: {tags_from_query}")
    query += " " + ", ".join(tags_from_query)
    print(f"Augmented query for search: {query}")

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


    # Apply score_email
    selected = rerank_full(query, selected)

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