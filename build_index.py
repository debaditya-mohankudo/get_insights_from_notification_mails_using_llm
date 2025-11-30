import os
import re
import mailbox
import pickle
from dataclasses import dataclass
from typing import List, Dict, Optional

import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


from email_models import EmailMessage

# ============================================================
#                 REGEX DEFINITIONS
# ============================================================

# ---- SUBJECT METADATA ----
PR_FROM_SUBJECT       = re.compile(r'\(PR\s*#(\d+)\)', re.IGNORECASE)
REPO_FROM_SUBJECT     = re.compile(r'\[([^\]]+)\]')
TICKET_FROM_SUBJECT   = re.compile(r'\b([A-Z]+-\d+)\b')

# ---- COMMITS ----
COMMIT_SIMPLE = re.compile(r'\b([0-9a-f]{7,40})\b(?:\s+(.+))?')

# ---- FILE CHANGES ----
FILES_GITHUB_PR = re.compile(
    r'^[\*\-\u2022]?\s*[MAD]\s+([^\s]+)(?:\s+\(\d+\))?',
    re.MULTILINE
)

FILES_PATTERN_1 = re.compile(
    r'(?:(modified|added|removed|deleted|renamed|changed):\s*)([^\s]+)',
    re.IGNORECASE
)

FILES_PATTERN_2 = re.compile(
    r'^[MAD]\s+([^\s]+)\s*(?:\(\d+\))?',
    re.MULTILINE
)

FILES_PATTERN_3 = re.compile(
    r'([^\s]+)\s*->\s*([^\s]+)'
)

CHANGE_COUNT = re.compile(r'\(([0-9]+)\)')


# ============================================================
#                 EXTRACTION HELPERS
# ============================================================

def extract_metadata_from_subject(subject: str) -> dict:
    repos = REPO_FROM_SUBJECT.findall(subject)
    pr_numbers = PR_FROM_SUBJECT.findall(subject)
    tickets = TICKET_FROM_SUBJECT.findall(subject)

    title = subject

    # Remove repo blocks
    if repos:
        for r in repos:
            title = title.replace(f'[{r}]', '')

    # Remove PR numbers
    if pr_numbers:
        for p in pr_numbers:
            title = title.replace(f'(PR #{p})', '')
            title = title.replace(f'(PR#{p})', '')

    # Remove tickets
    if tickets:
        for t in tickets:
            title = title.replace(t + '-', '')
            title = title.replace(t + ' ', '')

    return {
        "repos": repos or None,
        "pr_numbers": pr_numbers or None,
        "tickets": tickets or None,
        "pr_title": title.strip(" -_") or None,
    }


def extract_commits(text: str) -> Optional[List[str]]:
    commits = []
    for m in COMMIT_SIMPLE.finditer(text):
        sha = m.group(1)
        msg = m.group(2) or ""
        commits.append(f"{sha} {msg}".strip())
    return commits or None


def extract_files_modified(text: str) -> Optional[List[str]]:
    files = set()

    for m in FILES_GITHUB_PR.finditer(text):
        files.add(m.group(1).strip())

    for m in FILES_PATTERN_1.finditer(text):
        files.add(m.group(2).strip())

    for m in FILES_PATTERN_2.finditer(text):
        files.add(m.group(1).strip())

    for m in FILES_PATTERN_3.finditer(text):
        files.add(m.group(1).strip())
        files.add(m.group(2).strip())

    return list(files) or None


def extract_change_counts(text: str) -> Optional[Dict[str, int]]:
    result = {}

    for line in text.splitlines():
        m = FILES_GITHUB_PR.search(line) or FILES_PATTERN_2.search(line)
        if m:
            path = m.group(1)
            count = CHANGE_COUNT.search(line)
            if count:
                result[path] = int(count.group(1))

    return result or None




# ============================================================
#                 EMAIL EXTRACTION
# ============================================================

def extract_emails_from_mbox(mbox_path: str) -> List[EmailMessage]:
    print(f"Parsing: {mbox_path}")
    mbox = mailbox.mbox(mbox_path)
    emails = []

    for msg in mbox:
        subject = msg['subject'] or ""
        sender = msg['from'] or ""
        date = msg['date'] or ""

        if msg.is_multipart():
            body = ""
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_payload(decode=True).decode(errors="ignore")
        else:
            body = msg.get_payload(decode=True).decode(errors="ignore")

        meta = extract_metadata_from_subject(subject)

        commits = extract_commits(body)
        files = extract_files_modified(body)
        counts = extract_change_counts(body)

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
                change_counts=counts
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