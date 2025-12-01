import os
import re
import mailbox
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

from email_models import EmailMessage


# ============================================================
#                 REGEX DEFINITIONS
# ============================================================

PR_FROM_SUBJECT = re.compile(r'(?:PR\s*#|pull request\s*#|#)(\d+)', re.IGNORECASE)
REPO_FROM_SUBJECT = re.compile(r'\[([^\]]+)\]')
TICKET_FROM_SUBJECT = re.compile(r'\b([A-Z]+-\d+)\b')

COMMIT_SIMPLE = re.compile(
    r'^[ \t]*([0-9a-f]{7,40})\b(?:\s+(.+))?',
    re.MULTILINE
)

FILE_PATH = re.compile(
    r'^[ \t]*(?:M|A|D|R\d{1,3})\s+(?:a/|b/)?([A-Za-z0-9_./\-\+]+)',
    re.MULTILINE
)


# ============================================================
#                 EXTRACTION HELPERS
# ============================================================

def clean_html(html: str) -> str:
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n", strip=True)
    except Exception:
        return html


def extract_metadata_from_subject(subject: str) -> dict:
    repos = REPO_FROM_SUBJECT.findall(subject)
    pr_numbers = PR_FROM_SUBJECT.findall(subject)
    tickets = TICKET_FROM_SUBJECT.findall(subject)

    # Clean PR title
    clean_title = subject
    for r in repos:
        clean_title = clean_title.replace(f'[{r}]', '')

    for p in pr_numbers:
        clean_title = re.sub(rf'(PR\s*#|pull request\s*#|#){p}', '', clean_title, flags=re.I)

    for t in tickets:
        clean_title = re.sub(rf'\b{t}[:\-\s]*', '', clean_title)

    return {
        "repos": repos or None,
        "pr_numbers": pr_numbers or None,
        "tickets": tickets or None,
        "pr_title": clean_title.strip(" -:_") or None,
    }


def extract_commits(text: str) -> Optional[List[str]]:
    commits = [
        f"{m.group(1)} {m.group(2) or ''}".strip()
        for m in COMMIT_SIMPLE.finditer(text)
    ]
    return commits or None


def extract_files_modified(text: str) -> Optional[List[str]]:
    raw_matches = FILE_PATH.findall(text)
    cleaned = [
        re.sub(r'\(\d+\)$', '', m).strip()
        for m in raw_matches
    ]
    return list(set(cleaned)) or None


# ============================================================
#                 EMAIL BODY EXTRACTOR
# ============================================================

def extract_body(msg) -> str:
    """
    Extract best body:
    - Prefer text/plain
    - Fallback to stripped HTML
    """
    if not msg.is_multipart():
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode(errors='ignore')
        return str(payload)

    plain, html = [], []

    for part in msg.walk():
        ctype = part.get_content_type()
        payload = part.get_payload(decode=True)
        if payload is None:
            continue

        decoded = payload.decode(errors='ignore') if isinstance(payload, bytes) else str(payload)

        if ctype == "text/plain":
            plain.append(decoded)
        elif ctype == "text/html":
            html.append(clean_html(decoded))

    if plain:
        return "\n".join(plain)

    if html:
        return "\n".join(html)

    return ""


# ============================================================
#                 EMAIL EXTRACTION
# ============================================================

def extract_emails_from_mbox(mbox_path: str) -> List[EmailMessage]:
    print(f"📦 Parsing: {mbox_path}")

    mbox = mailbox.mbox(mbox_path)
    results = []

    for msg in tqdm(mbox, desc="Extracting emails"):
        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        date = msg.get("date", "")

        body = extract_body(msg)
        meta = extract_metadata_from_subject(subject)

        results.append(
            EmailMessage(
                subject=subject,
                sender=sender,
                date=date,
                body=body,

                pr_numbers=meta["pr_numbers"],
                repos=meta["repos"],
                tickets=meta["tickets"],
                pr_title=meta["pr_title"],

                commits=extract_commits(body),
                files_modified=extract_files_modified(body),
            )
        )

    return results


# ============================================================
#               INDEX BUILDER
# ============================================================

def embed_and_index(emails: List[EmailMessage], index_path="index.faiss", meta_path="meta.pkl"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [email.full_text() for email in emails]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200

    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(emails, f)

    print(f"✔ FAISS index saved: {index_path}")
    print(f"✔ Metadata saved: {meta_path}")


# ============================================================
#               MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    ROOT = Path("/Users/debaditya/workspace/trash_mails")

    emails: List[EmailMessage] = []

    for item in ROOT.iterdir():
        # Only parse folder/*.mbox/mbox
        if item.suffix == ".mbox":
            mbox_path = item / "mbox"
            if mbox_path.exists():
                emails.extend(extract_emails_from_mbox(str(mbox_path)))

    print(f"📨 Total emails collected: {len(emails)}")

    embed_and_index(emails)