import os
import re
import mailbox
import pickle
from typing import List, Optional

import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from bs4 import BeautifulSoup
from email_models import EmailMessage


# ============================================================
#                 REGEX DEFINITIONS (FIXED)
# ============================================================

# PR patterns: (#1234), PR #1234, pull request #1234
PR_FROM_SUBJECT = re.compile(
    r'(?:PR\s*#|pull request\s*#|#)(\d+)',
    re.IGNORECASE
)

REPO_FROM_SUBJECT = re.compile(r'\[([^\]]+)\]')

TICKET_FROM_SUBJECT = re.compile(r'\b([A-Z]+-\d+)\b')

# Commits (SHA)
COMMIT_SIMPLE = re.compile(
    r'^[ \t]*([0-9a-f]{7,40})\b(?:\s+(.+))?',
    re.MULTILINE
)

# FILE CHANGES — FIXED for GitHub email format:
# "    M path/to/file.php (1)"
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
        return soup.get_text("\n")
    except:
        return html


def extract_metadata_from_subject(subject: str) -> dict:
    repos = REPO_FROM_SUBJECT.findall(subject)
    pr_numbers = PR_FROM_SUBJECT.findall(subject)
    tickets = TICKET_FROM_SUBJECT.findall(subject)

    title = subject

    for r in repos:
        title = title.replace(f'[{r}]', '')

    for p in pr_numbers:
        title = re.sub(rf'(PR\s*#|pull request\s*#|#){p}', '', title, flags=re.I)

    for t in tickets:
        title = re.sub(rf'\b{t}[:\-\s]*', '', title)

    return {
        "repos": repos or None,
        "pr_numbers": pr_numbers or None,
        "tickets": tickets or None,
        "pr_title": title.strip(" -:_") or None,
    }


def extract_commits(text: str) -> Optional[List[str]]:
    commits = []
    for m in COMMIT_SIMPLE.finditer(text):
        sha = m.group(1)
        msg = m.group(2) or ""
        commits.append(f"{sha} {msg}".strip())
    return commits or None


def extract_files_modified(text: str):
    """EXACT FIX: Handles 'M file.php (1)' and strips '(1)'."""
    matches = FILE_PATH.findall(text)

    cleaned = []
    for m in matches:
        # Remove trailing (1), (2), etc.
        m = re.sub(r'\(\d+\)$', '', m).strip()
        cleaned.append(m)

    return list(set(cleaned)) or None


# ============================================================
#                 EMAIL EXTRACTION
# ============================================================

def extract_emails_from_mbox(mbox_path: str) -> List[EmailMessage]:
    print(f"Parsing: {mbox_path}")
    mbox = mailbox.mbox(mbox_path)
    emails = []

    for msg in mbox:
        subject = msg.get('subject', "")
        sender = msg.get('from', "")
        date = msg.get('date', "")

        # Extract body (Prefer text/plain but fallback to HTML)
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                payload = part.get_payload(decode=True)

                if payload is None:
                    continue

                decoded = payload.decode(errors='ignore') if isinstance(payload, bytes) else str(payload)

                if ctype == "text/plain":
                    body += decoded
                elif ctype == "text/html" and not body.strip():
                    body = clean_html(decoded)
        else:
            payload = msg.get_payload(decode=True)
            body = payload.decode(errors='ignore') if isinstance(payload, bytes) else str(payload)

        meta = extract_metadata_from_subject(subject)
        commits = extract_commits(body)
        files = extract_files_modified(body)

        emails.append(
            EmailMessage(
                subject=subject,
                sender=sender,
                date=date,
                body=body,

                pr_numbers=meta["pr_numbers"],
                repos=meta["repos"],
                tickets=meta["tickets"],
                pr_title=meta["pr_title"],

                commits=commits,
                files_modified=files,
            )
        )

    return emails


# ============================================================
#               INDEX BUILDER
# ============================================================

def embed_and_index(emails: List[EmailMessage]):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [email.full_text() for email in emails]

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200

    index.add(embeddings)

    faiss.write_index(index, "index.faiss")
    with open("meta.pkl", "wb") as f:
        pickle.dump(emails, f)

    print("FAISS index saved as index.faiss")
    print("Metadata saved as meta.pkl")


# ============================================================
#               MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    ROOT = "/Users/debaditya/workspace/trash_mails"

    emails = []

    for folder in os.listdir(ROOT):
        if folder.endswith(".mbox"):
            mbox_path = os.path.join(ROOT, folder, "mbox")
            if os.path.exists(mbox_path):
                emails.extend(extract_emails_from_mbox(mbox_path))

    print(f"Total emails collected: {len(emails)}")

    embed_and_index(emails)