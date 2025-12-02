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
    extract_body,
    extract_pr_from_message_id,
)
import mailbox

class EmailExtractor:
    def extract_emails_from_mbox(self, mbox_path: str) -> List[EmailMessage]:
        """Fast, allocation-optimized mbox → EmailMessage parser."""

        print(f"📦 Parsing: {mbox_path}")

        mbox = mailbox.mbox(mbox_path)

        # Pre-bind functions for speed
        _extract_body = extract_body
        _extract_md = extract_markdown_sections
        _extract_meta = extract_metadata_from_subject
        _extract_commits = extract_commits
        _extract_files = extract_files_modified
        _extract_pr_from_msgid = extract_pr_from_message_id

        EmailMsg = EmailMessage

        results = []
        append = results.append

        for msg in mbox:
            subject = msg.get("subject", "")
            sender = msg.get("from", "")
            date = msg.get("date", "")
            message_id = msg.get("Message-ID", "") or ""

            body = _extract_body(msg)
            meta = _extract_meta(subject)

            # -----------------------------
            #  PR NUMBERS (Fix: must be int)
            # -----------------------------
            pr_subject_list = meta["pr_numbers"]
            # PR via Message-ID
            pr_from_msgid = _extract_pr_from_msgid(message_id)

            # Final PR list
            pr_numbers = pr_subject_list or []
            if pr_from_msgid and pr_from_msgid not in pr_numbers:
                pr_numbers.append(pr_from_msgid)

            # -----------------------------
            #  Build EmailMessage object
            # -----------------------------
            append(
                EmailMsg(
                    subject=subject,
                    sender=sender,
                    date=date,

                    message_id=message_id,     # NEW FIELD
                    pr_numbers=pr_numbers or None,

                    repos=meta["repos"],
                    tickets=meta["tickets"],
                    pr_title=meta["pr_title"],

                    body=body,
                    markdown= _extract_md(body),

                    commits=_extract_commits(body),
                    files_modified=_extract_files(body),
                )
            )

        return results
