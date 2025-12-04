# ============================================================
#                 EMAIL EXTRACTION (OPTIMIZED)
# ============================================================

from typing import List
from _internal.email_models import EmailMessage
from _internal.markdown_sections import extract_heading_sections_with_content, extract_markdown_sections
from _internal.helpers import (
    extract_metadata_from_subject,
    extract_commits,
    extract_files_modified,
    extract_body,
    extract_pr_from_message_id,
)
import mailbox
from _internal.tag_classifier import classify_tags
from _internal.tags_from_file import classify_tags_from_files
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
        _classify_tags = classify_tags
        _classify_tags_from_files = classify_tags_from_files

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
            pr_numbers = [int(p) for p in pr_numbers] if pr_numbers else None
            pr_numbers = list(set(pr_numbers)) if pr_numbers else None

            files_modified = _extract_files(body)
            markdown_sections = _extract_md(body)
            sections = extract_heading_sections_with_content(body)

            tags_from_title = _classify_tags(meta["pr_title"])
            tags_from_files = _classify_tags_from_files(files_modified or [])
            tags_from_section = _classify_tags(','.join(','.join(s[1]) for s in sections))

            combined_tags = sorted(set(tags_from_title) | set(tags_from_files) | set(tags_from_section))
            #if  pr_numbers is not None:
            #    #print(tags_from_section, sections, pr_numbers)
            # -----------------------------
            #  Build EmailMessage object
            # -----------------------------
            append(
                EmailMsg(
                    subject=subject,
                    date=date,

                    message_id=message_id,     # NEW FIELD
                    pr_numbers=pr_numbers or None,

                    repos=meta["repos"],
                    tickets=meta["tickets"],
                    pr_title=meta["pr_title"],
                    contributors=meta["contributors"],
                    tags=combined_tags,

                    body=body,
                    markdown= markdown_sections,

                    commits=_extract_commits(body),
                    files_modified=files_modified,
                )
            )

        return results
