import pickle
import sys
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import ollama

from email_models import EmailMessage


# ============================================================
#                    LOAD INDEX + METADATA
# ============================================================

with open("meta.pkl", "rb") as f:
    META: List[EmailMessage] = pickle.load(f)

# assert isinstance(META[0], EmailMessage), "META does not contain EmailMessage objects"

INDEX = faiss.read_index("index.faiss")
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TOP_K = 5


# ============================================================
#                    NORMALIZATION UTILITIES
# ============================================================

def normalize(text: str) -> str:
    return text.lower().replace("#", "").strip()


def extract_pr_number(query: str) -> Optional[str]:
    digits = ''.join([c for c in query if c.isdigit()])
    return digits if digits else None


# ============================================================
#           IMPROVED EXACT MATCH WEIGHTED SEARCH
# ============================================================

def find_exact_matches(query: str, top_k: int = TOP_K) -> List[int]:
    q = normalize(query)
    tokens = q.split()
    scores: dict[int, int] = {}

    def add(idx: int, amt: int):
        scores[idx] = scores.get(idx, 0) + amt

    for idx, email in enumerate(META):

        # PR numbers
        if email.pr_numbers:
            for pr in email.pr_numbers:
                p = normalize(pr)
                if p in q or q in p:
                    add(idx, 200)
                if any(t in p for t in tokens):
                    add(idx, 100)

        # Repo names
        if email.repos:
            for r in email.repos:
                rn = normalize(r)
                if rn in q:
                    add(idx, 70)
                if any(tok in rn for tok in tokens):
                    add(idx, 40)

        # Tickets
        if email.tickets:
            for t in email.tickets:
                tn = normalize(t)
                if tn in q:
                    add(idx, 70)
                if any(tok in tn for tok in tokens):
                    add(idx, 40)

        # Commits
        if email.commits:
            for c in email.commits:
                sha = normalize(c.split()[0])
                if sha.startswith(q) or q.startswith(sha):
                    add(idx, 90)
                if any(tok in sha for tok in tokens):
                    add(idx, 40)

        # File paths
        if email.files_modified:
            for f in email.files_modified:
                fn = normalize(f)
                if fn in q or q in fn:
                    add(idx, 50)
                if any(tok in fn for tok in tokens):
                    add(idx, 25)

        # PR title
        if email.pr_title:
            t = normalize(email.pr_title)
            if t in q:
                add(idx, 50)
            if any(tok in t for tok in tokens):
                add(idx, 20)

        # Body
        body = normalize(email.body or "")
        if q in body:
            add(idx, 10)
        if any(tok in body for tok in tokens):
            add(idx, 5)
            
        # Markdown sections
        if email.markdown:
            for section in email.markdown.values():
                if isinstance(section, list):
                    for item in section:
                        item_text = normalize(str(item))
                        if q in item_text:
                            add(idx, 10)
                        if any(tok in item_text for tok in tokens):
                            add(idx, 5)

    if not scores:
        return []

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, score in ranked[:top_k]]


# ============================================================
#           STRICT FILTER FOR EXACT PR MATCHES
# ============================================================

def strict_filter_by_pr(indices: List[int], query: str) -> List[int]:
    pr = extract_pr_number(query)
    if not pr:
        return indices

    filtered = []
    for idx in indices:
        email = META[idx]
        if email.pr_numbers and pr in email.pr_numbers:
            filtered.append(idx)

    return filtered or indices


# ============================================================
#                    SEMANTIC SEARCH FALLBACK
# ============================================================

def semantic_search(query: str, k: int = TOP_K) -> List[int]:
    extras: List[str] = []

    for email in META:
        if email.repos: extras.extend(email.repos)
        if email.pr_numbers: extras.extend(email.pr_numbers)
        if email.tickets: extras.extend(email.tickets)
        if email.pr_title: extras.append(email.pr_title)

    extras = list(set(normalize(x) for x in extras if x))
    expanded_query = query + " " + " ".join(extras[:30])

    emb = MODEL.encode([expanded_query])
    _, idxs = INDEX.search(emb, k)
    return idxs[0].tolist()


# ============================================================
#               ENHANCED NON-LLM PR SUMMARY
# ============================================================

def enhanced_pr_summary(indices: List[int], pr: str):
    """Prints detailed structured information about a PR without LLM."""

    print(f"\n==================== PR {pr} — Summary ====================\n")

    if not indices:
        print(f"No emails found for PR {pr}.")
        return

    all_commits = []
    all_files = []
    all_titles = set()
    timeline = []

    for idx in indices:
        email = META[idx]

        if email.pr_title:
            all_titles.add(email.pr_title.strip())

        if email.commits:
            all_commits.extend(email.commits)

        if email.files_modified:
            all_files.extend(email.files_modified)

        timeline.append((email.date, idx, email.subject))

    timeline.sort()

    unique_commits = sorted(set(all_commits))
    unique_files = sorted(set(all_files))

    print(f"PR Title      : {next(iter(all_titles)) if all_titles else '(none)'}")
    print(f"Commits Found : {len(unique_commits)}")
    print(f"Files Changed : {len(unique_files)}")
    print(f"Emails Found  : {len(indices)}\n")

    print("----- Commits -----")
    for c in unique_commits:
        print("  -", c)
    print()

    print("----- Files Modified -----")
    for f in unique_files:
        print("  -", f)
    print()

    print("----- Timeline -----")
    for dt, idx, subject in timeline:
        print(f"{dt}  |  {subject}  (email #{idx})")
    print()

    print("================== END OF PR SUMMARY ==================\n")


# ============================================================
#            FORMAT EMAIL CHUNKS FOR LLM ANSWERING
# ============================================================

def get_email_chunks(indices: List[int], query: str) -> List[str]:
    chunks: List[str] = []
    pr = extract_pr_number(query)

    for idx in indices:
        email: EmailMessage = META[idx]

        block = f"""
========================================
MATCHED EMAIL INDEX: {idx}
Email PR Numbers: {email.pr_numbers}
Email Repo: {email.repos}
Email Tickets: {email.tickets}
Email PR Title: {email.pr_title}
========================================

Subject: {email.subject}
From: {email.sender}
Date: {email.date}

Commits:
{email.commits}

Files Modified:
{email.files_modified}


Body:
{email.body[:1500]}
"""
        chunks.append(block)

    return chunks


# ============================================================
#                      ASK LOCAL LLM
# ============================================================

def ask_llm(query: str, chunks: List[str], pr: Optional[str]) -> str:
    prompt = f"""
You are an expert GitHub PR analyst.

Use ONLY the information in the chunks below.

{'-'*60}
{chr(10).join(chunks)}
{'-'*60}

User Query: "{query}"
"""

    response = ollama.generate(
        model="llama3.2:3b",
        prompt=prompt
    )
    return response["response"]


# ============================================================
#                      MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python query_llm.py \"your question\"")
        sys.exit()

    query: str = sys.argv[1]
    pr: Optional[str] = extract_pr_number(query)

    # ------------------------------------------------------------
    # CASE 1: PR MODE (FAST, NO LLM)
    # ------------------------------------------------------------
    if pr:
        print(f"[PR mode activated → PR #{pr}]")

        exact = find_exact_matches(query)
        indices = strict_filter_by_pr(exact, query)

        if not indices:
            semantic = semantic_search(query)
            indices = strict_filter_by_pr(semantic, query)

        enhanced_pr_summary(indices, pr)
        sys.exit()

    # ------------------------------------------------------------
    # CASE 2: GENERAL QUESTION → USE LLM
    # ------------------------------------------------------------
    print("[General question → using LLM]")

    exact = find_exact_matches(query)

    if exact:
        print(f"[Exact-match retrieval → {len(exact)} results]")
        indices = exact[:TOP_K]
    else:
        print("[Semantic retrieval]")
        indices = semantic_search(query)

    chunks = get_email_chunks(indices, query)
    answer = ask_llm(query, chunks, pr=None)

    print("\n==================================================")
    print(answer)
    print("==================================================\n")
