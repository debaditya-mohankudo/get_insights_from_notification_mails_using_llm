"""
helpers.py

Utility functions for extracting metadata from GitHub notification emails.
This module provides:

• Regex definitions for PR numbers, tickets, repos, commit lines, and file paths.
• Robust extraction of plain-text and HTML email bodies.
• Title cleanup and metadata parsing from email subject lines.
• Commit and file-diff extraction suitable for indexing or analysis.

These helpers are used during indexing of exported Apple Mail `.mbox` files.
All functions are designed to tolerate the noisy, inconsistent structure common
in GitHub notification email formats.
"""

import re
from bs4 import BeautifulSoup
from typing import List, Optional

from collections import namedtuple
CommitInfo = namedtuple("CommitInfo", ["sha", "short", "message"])

# ============================================================
#                 REGEX DEFINITIONS
# ============================================================

# PR references in subject lines:
# Matches formats like:
#   "PR #8040", "pull request #8040", "#8040"
PR_FROM_SUBJECT = re.compile(
    r'(?:PR\s*#|pull request\s*#|#)(\d+)', 
    re.IGNORECASE
)

# Repository names appear inside square brackets in subjects:
#   "[fuzzycert/fuzzycert_codecops]"
REPO_FROM_SUBJECT = re.compile(r'\[([^\]]+)\]')

# Ticket identifiers such as "FIZZY-2044" or "XY-12345"
TICKET_FROM_SUBJECT = re.compile(r'\b([A-Z]+-\d+)\b')

# Commit lines in text/plain emails.
# Matches a SHA (7–40 hex chars) at start of line + optional message text.
COMMIT_SIMPLE = re.compile(
    r'^[ \t]*([0-9a-f]{7,40})\b(?:\s+(.+))?',
    re.MULTILINE
)

# File paths from Git diff headers:
# Supports:
#   M file.js
#   A src/core/index.py
#   D old/module.c
#   R100 a/path b/path
FILE_PATH = re.compile(
    r'^[ \t]*(?:M|A|D|R\d{1,3})\s+(?:a/|b/)?([A-Za-z0-9_./\-\+]+)',
    re.MULTILINE
)

MENTION_RE = re.compile(r'@([A-Za-z0-9-]+)')
def extract_contributors(body: str):
    """
    Extract GitHub contributor usernames from an email body.
    Returns a unique unordered list.
    """
    if not body:
        return []

    matches = MENTION_RE.findall(body)
    return list(set(matches))
# ============================================================
#                 EMAIL BODY EXTRACTOR
# ============================================================

def extract_body(msg) -> str:
    """
    Extract the most useful body text from an email message.

    Priority:
        1. text/plain parts
        2. text/html parts (converted to plain text)
        3. fallback: raw payload as string

    This supports real-world GitHub notification emails which often contain
    mixed multipart structures.

    Parameters:
        msg: email.message.Message (from mailbox.mbox or similar)

    Returns:
        str: Best-effort clean body text.
    """
    # If message is not multipart, decode directly
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

        # Decode safely
        decoded = payload.decode(errors='ignore') if isinstance(payload, bytes) else str(payload)

        if ctype == "text/plain":
            plain.append(decoded)

        elif ctype == "text/html":
            html.append(clean_html(decoded))

    # Prefer plain text
    if plain:
        return "\n".join(plain)

    # Fallback to cleaned HTML
    if html:
        return "\n".join(html)

    return ""


def clean_html(html: str) -> str:
    """
    Convert HTML content into clean plain text using BeautifulSoup.

    Parameters:
        html (str): HTML string.

    Returns:
        str: Extracted plain text. If parsing fails, the raw HTML is returned.
    """
    try:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text("\n", strip=True)
    except Exception:
        return html


# ============================================================
#           SUBJECT METADATA EXTRACTOR
# ============================================================

def extract_metadata_from_subject(subject: str) -> dict:
    """
    Parse repositories, PR numbers, tickets, and derive a cleaned PR title
    from the raw email subject line.

    Subject examples:
        "[repo/name] PR #8040: Fix bug DIGI-2044"
        "[service] #5521 - Update handler"

    Extraction steps:
        • repos: items inside [...]
        • pr_numbers: PR #, pull request #, or #NNNN
        • tickets: DIGI-xxxx, CC-xxxx, etc.
        • pr_title: subject with repos, PR numbers, and tickets removed

    Parameters:
        subject (str): raw subject header.

    Returns:
        dict: {
            "repos": List[str] or None,
            "pr_numbers": List[str] or None,
            "tickets": List[str] or None,
            "pr_title": str or None
        }
    """
    repos = REPO_FROM_SUBJECT.findall(subject)
    pr_numbers = PR_FROM_SUBJECT.findall(subject)
    tickets = TICKET_FROM_SUBJECT.findall(subject)
    contributors = extract_contributors(subject)

    # Begin with original subject and remove extracted metadata
    clean_title = subject

    # Remove repo brackets: [repo/name]
    for r in repos:
        clean_title = clean_title.replace(f'[{r}]', '')

    # Remove PR markers: PR #NNNN, pull request #NNNN, or #NNNN
    for p in pr_numbers:
        clean_title = re.sub(
            rf'(PR\s*#|pull request\s*#|#){p}', 
            '', 
            clean_title, 
            flags=re.I
        )

    # Remove ticket IDs
    for t in tickets:
        clean_title = re.sub(rf'\b{t}[:\-\s]*', '', clean_title)

    return {
        "repos": repos or None,
        "pr_numbers": pr_numbers or None,
        "tickets": tickets or None,
        "pr_title": clean_title.strip(" -:_") or None,
        "contributors": contributors or None,
    }


# ============================================================
#           COMMIT + FILE EXTRACTION HELPERS
# ============================================================

def extract_commits(text: str) -> Optional[List[CommitInfo]]:
    """
    Extract commit entries from email body text.

    Each match captures:
        • SHA (7–40 hex characters)
        • Optional commit message

    Output format:
        ["<sha> <message>", "<sha>", ...]

    Parameters:
        text (str): email body text.

    Returns:
        list[str] or None: commit entries, or None if no commits found.
    """
    commits = [
        f"{m.group(1)} {m.group(2) or ''}".strip()
        for m in COMMIT_SIMPLE.finditer(text)
    ]
    return parse_commit_lines(commits) or None
 


def extract_files_modified(text: str) -> Optional[List[str]]:
    """
    Extract modified file paths from a Git-style diff snippet.

    Matches diff lines such as:
        M file.json
        A src/core/module.js
        D old/file.txt
        R100 a/file.py b/file.py

    Post-processing:
        • Removes trailing "(number)" metadata sometimes found in diff outputs
        • Deduplicates the list

    Parameters:
        text (str): email body text.

    Returns:
        list[str] or None: modified file paths, or None if none found.
    """
    raw_matches = FILE_PATH.findall(text)

    cleaned = [
        re.sub(r'\(\d+\)$', '', m).strip()
        for m in raw_matches
    ]

    flat_parts = [p for path in cleaned for p in path.split("/")]
    files_modified = list(set(flat_parts)) if flat_parts else []

    return files_modified

def generate_tags_from_pr_title(pr_title: str):
    """
    Generate classification tags based on keywords found in a PR title.

    Rules:
      - If "bug" or "fix" → add ["bug", "fix"]
      - If "sql" / "table" / "database" / "db" → add ["sql", "database"]
      - If "ui" → add ["ui"]
      - If "api" → add ["api"]
      - If "performance" → add ["performance"]
      - If "security" / "vulnerability" → add ["security"]
      - If "refactor" / "cleanup" → add ["refactor"]
      - If "auth" / "login" / "oauth" → add ["authentication", "auth"]

    Returns:
        List[str] of unique tags.
    """
    if not pr_title:
        return []

    title = pr_title.lower()
    tags = []

    # bug / fix detection
    if "bug" in title or "fix" in title:
        tags.extend(["bug", "fix"])

    # database related
    if any(word in title for word in ["sql", "table", "database", "db"]):
        tags.extend(["sql", "database"])

    # ui
    if "ui" in title:
        tags.append("ui")

    # api
    if "api" in title:
        tags.append("api")

    # performance
    if "performance" in title:
        tags.append("performance")
    
    # security
    if "security" in title or "vulnerability" in title:
        tags.append("security")
    
    # refactor / cleanup
    if "refactor" in title or "cleanup" in title:
        tags.append("refactor")
    
    # authentication
    if any(word in title for word in ["auth", "login", "oauth"]):
        tags.extend(["authentication", "auth"])

    return list(set(tags))  # remove duplicates


def extract_pr_from_message_id(msgid: str) -> int | None:
    if not msgid:
        return None
    PR_MSGID_RE = re.compile(r"/pull/(\d+)/")
    m = PR_MSGID_RE.search(msgid)
    if m:
        return int(m.group(1))
    return None

def extract_prs_from_body_links(body: str) -> list[int]:
    if not body:
        return []
    PR_LINK_RE = re.compile(r"https?://github\.com/[^/]+/[^/]+/pull/(\d+)", re.IGNORECASE)
    matches = PR_LINK_RE.findall(body)
    return [int(m) for m in matches]

def extract_tickets_from_body(body: str) -> list[str]:
    if not body:
        return []
    TICKET_BODY_RE = re.compile(r"\b([A-Z][A-Z0-9]{1,10}-\d{1,6})\b")
    return TICKET_BODY_RE.findall(body)



def parse_commit_lines(raw_commits: List[str]) -> List[CommitInfo]:
    commits = []
    for line in raw_commits:
        sha, msg = line.split(" ", 1)
        commits.append(
            CommitInfo(
                sha=sha,
                short=sha[:7],
                message=msg.strip()
            )
        )
    return commits
