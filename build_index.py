from __future__ import annotations

import os
import pickle
import mailbox
import re
from dataclasses import dataclass
from typing import List

import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
BASE_DIR: str = "/Users/debaditya/workspace/trash_mails"
EMBED_MODEL: str = "all-MiniLM-L6-v2"


# ------------------------------------------------------
# DATACLASS
# ------------------------------------------------------
@dataclass
class EmailMessage:
    subject: str
    sender: str
    date: str
    body: str

    # Extended extracted fields
    commits: List[str] | None = None
    pr_numbers: List[str] | None = None
    repos: List[str] | None = None
    issues: List[str] | None = None
    sql_queries: List[str] | None = None
    files_modified: List[str] | None = None   # ← NEW FIELD
    tags: List[str] | None = None

    def full_text(self) -> str:
        """
        Text used for embeddings.
        Includes structured metadata for richer vector understanding.
        """
        extra = []

        if self.commits:
            extra.append("Commits:\n" + "\n".join(self.commits))
        if self.pr_numbers:
            extra.append("PRs:\n" + ", ".join(self.pr_numbers))
        if self.repos:
            extra.append("Repos:\n" + ", ".join(self.repos))
        if self.issues:
            extra.append("Issues:\n" + ", ".join(self.issues))
        if self.sql_queries:
            extra.append("SQL:\n" + "\n".join(self.sql_queries))
        if self.files_modified:
            extra.append("Files Modified:\n" + "\n".join(self.files_modified))

        return f"{self.subject}\n\n{self.body}\n\n" + "\n\n".join(extra)


# ------------------------------------------------------
# REGEX PATTERNS
# ------------------------------------------------------
COMMIT_REGEX = re.compile(
    r'(?:^|\n)[\*\-\•]?\s*([a-f0-9]{7,40})\s*[-:]\s*(.+)',
    re.IGNORECASE
)

PR_REGEX = re.compile(
    r'(?:PR|Pull Request)\s*#?(\d+)',
    re.IGNORECASE
)

REPO_REGEX = re.compile(
    r'github\.com/([\w\-_]+/[\w\-_]+)',
    re.IGNORECASE
)

ISSUE_REGEX = re.compile(
    r'(?:Issue|Fixes|Closes)\s*#?(\d+)',
    re.IGNORECASE
)

SQL_REGEX = re.compile(
    r'((SELECT|UPDATE|DELETE|INSERT|CREATE)[\s\S]+?;)',
    re.IGNORECASE
)

FILES_REGEX = re.compile(
    r'(?:(modified|added|removed|deleted|renamed|changed):\s*)([\w\./\-\_]+)',
    re.IGNORECASE
)
# ------------------------------------------------------
# EXTRACTION HELPERS
# ------------------------------------------------------
def extract_commits(text: str) -> List[str]:
    commits = []
    for match in COMMIT_REGEX.finditer(text):
        sha = match.group(1)
        msg = match.group(2).strip()
        commits.append(f"{sha} - {msg}")
    return commits


def extract_pr_numbers(text: str) -> List[str]:
    return list({m.group(1) for m in PR_REGEX.finditer(text)})


def extract_repos(text: str) -> List[str]:
    return list({m.group(1) for m in REPO_REGEX.finditer(text)})


def extract_issues(text: str) -> List[str]:
    return list({m.group(1) for m in ISSUE_REGEX.finditer(text)})


def extract_sql(text: str) -> List[str]:
    return [m.group(1).strip() for m in SQL_REGEX.finditer(text)]

def extract_files_modified(text: str) -> List[str]:
    """
    Extract filenames from GitHub-style change summaries.
    """
    results = []
    for match in FILES_REGEX.finditer(text):
        file = match.group(2).strip()
        results.append(file)
    return results

def detect_tags(email: EmailMessage) -> List[str] | None:
    tags = []

    if "github" in email.sender.lower():
        tags.append("github")
    if email.commits:
        tags.append("commit")
    if email.pr_numbers:
        tags.append("pr")
    if email.issues:
        tags.append("issue")
    if email.repos:
        tags.append("repo")
    if email.sql_queries:
        tags.append("sql")
    if "security" in email.subject.lower():
        tags.append("security")
    if "otp" in email.body.lower():
        tags.append("otp")
    if email.files_modified:
        tags.append("files")

    return tags or None


# ------------------------------------------------------
# EMAIL PARSER
# ------------------------------------------------------
def extract_emails(mbox_path: str) -> List[EmailMessage]:
    """
    Extract EmailMessage dataclasses from a .mbox file.
    """
    mbox = mailbox.mbox(mbox_path)
    messages: List[EmailMessage] = []

    for msg in mbox:
        try:
            subject = msg.get("subject", "").strip()
            sender = msg.get("from", "").strip()
            date = msg.get("date", "").strip()

            # extract body
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        try:
                            body_part = (
                                part.get_payload(decode=True)
                                .decode(errors="ignore")
                            )
                            body += body_part
                        except Exception:
                            pass
            else:
                try:
                    body = msg.get_payload(decode=True).decode(errors="ignore")
                except Exception:
                    body = ""

            # run extractors
            commits = extract_commits(body)
            prs = extract_pr_numbers(body)
            repos = extract_repos(body)
            issues = extract_issues(body)
            sqls = extract_sql(body)
            files = extract_files_modified(body)

            email_obj = EmailMessage(
                subject=subject,
                sender=sender,
                date=date,
                body=body,
                commits=commits or None,
                pr_numbers=prs or None,
                repos=repos or None,
                issues=issues or None,
                sql_queries=sqls or None,
                files_modified=files or None,   # ← NEW
            )

            email_obj.tags = detect_tags(email_obj)

            messages.append(email_obj)

        except Exception as e:
            print(f"Error parsing email: {e}")

    return messages


# ------------------------------------------------------
# LOAD ALL MAILBOXES
# ------------------------------------------------------
def load_all_mails() -> List[EmailMessage]:
    all_messages: List[EmailMessage] = []

    for root, dirs, files in os.walk(BASE_DIR):
        for f in files:
            if f == "mbox":
                path = os.path.join(root, f)
                print("Parsing:", path)
                msgs = extract_emails(path)
                all_messages.extend(msgs)

    return all_messages


# ------------------------------------------------------
# EMBEDDINGS + FAISS INDEX
# ------------------------------------------------------
def embed_and_index(messages: List[EmailMessage]) -> None:
    model = SentenceTransformer(EMBED_MODEL)

    texts = [m.full_text() for m in messages]

    print("Embedding emails...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1] # embedding dimension
    index = faiss.IndexFlatL2(dim) # good enough for 5 million vectors
    index.add(embeddings)

    faiss.write_index(index, "index.faiss")
    print("FAISS index saved as index.faiss")

    with open("meta.pkl", "wb") as f:
        pickle.dump(messages, f)

    print("Metadata saved as meta.pkl")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    mails = load_all_mails()
    print("Total emails collected:", len(mails))
    embed_and_index(mails)

