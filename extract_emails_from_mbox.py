# ============================================================
#                 EMAIL EXTRACTION
# ============================================================
from pyparsing import List
from email_models import EmailMessage
from markdown_sections import extract_markdown_sections
from helpers import extract_metadata_from_subject, extract_commits, extract_files_modified, extract_body
import mailbox
from tqdm import tqdm

class EmailExtractor:
    def extract_emails_from_mbox(self, mbox_path: str) -> List[EmailMessage]:
        print(f"📦 Parsing: {mbox_path}")

        mbox = mailbox.mbox(mbox_path)
        results = []
        
        for msg in tqdm(mbox, desc="Extracting emails"):
            subject = msg.get("subject", "")
            sender = msg.get("from", "")
            date = msg.get("date", "")

            body = extract_body(msg)
            meta = extract_metadata_from_subject(subject)
            md = extract_markdown_sections(body)
            results.append(
                EmailMessage(
                    subject=subject,
                    sender=sender,
                    date=date,
                    body=body,
                    markdown=md,

                    pr_numbers=meta["pr_numbers"],
                    repos=meta["repos"],
                    tickets=meta["tickets"],
                    pr_title=meta["pr_title"],

                    commits=extract_commits(body),
                    files_modified=extract_files_modified(body),
                )
            )

        return results

