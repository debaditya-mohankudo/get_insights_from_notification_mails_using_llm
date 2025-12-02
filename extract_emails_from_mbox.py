# ============================================================
#                 EMAIL EXTRACTION (OPTIMIZED)
# ============================================================

from typing import List
from email_models import EmailMessage
from markdown_sections import extract_markdown_sections
from helpers import (
    extract_metadata_from_subject,
    extract_commits,
    extract_files_modified,
    extract_body
)
import mailbox

class EmailExtractor:
    def extract_emails_from_mbox(self, mbox_path: str) -> List[EmailMessage]:
        """Fast, allocation-optimized mbox → EmailMessage parser."""

        print(f"📦 Parsing: {mbox_path}")

        mbox = mailbox.mbox(mbox_path)

        # Pre-bind functions for faster lookup in loops
        _extract_body = extract_body
        _extract_md = extract_markdown_sections
        _extract_meta = extract_metadata_from_subject
        _extract_commits = extract_commits
        _extract_files = extract_files_modified
        EmailMsg = EmailMessage

        results = []
        append_result = results.append  # local binding → 2x faster in tight loops

        # TQDM is optional; it slows down multiprocessing
        # Remove TQDM entirely inside parallel workers if needed.
        for msg in mbox:
            subject = msg.get("subject", "")
            sender = msg.get("from", "")
            date = msg.get("date", "")

            body = _extract_body(msg)
            meta = _extract_meta(subject)

            # Avoid dict lookups per key by unpacking once
            pr_numbers = meta["pr_numbers"]
            repos = meta["repos"]
            tickets = meta["tickets"]
            pr_title = meta["pr_title"]

            append_result(
                EmailMsg(
                    subject=subject,
                    sender=sender,
                    date=date,
                    body=body,
                    markdown=_extract_md(body),

                    pr_numbers=pr_numbers,
                    repos=repos,
                    tickets=tickets,
                    pr_title=pr_title,

                    commits=_extract_commits(body),
                    files_modified=_extract_files(body),
                )
            )

        return results