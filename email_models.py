from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class EmailMessage:
    subject: str
    sender: str
    date: str
    body: str

    pr_numbers: Optional[List[str]] = None
    repos: Optional[List[str]] = None
    tickets: Optional[List[str]] = None
    pr_title: Optional[str] = None
    markdown: Optional[Dict] = None

    commits: Optional[List[str]] = None
    files_modified: Optional[List[str]] = None
    change_counts: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        # Normalize commit SHAs to first 7 chars
        if self.commits:
            normalized = []
            for c in self.commits:
                parts = c.split(" ", 1)
                sha = parts[0][:7]                # <-- only first 7 chars
                msg = parts[1] if len(parts) > 1 else ""
                normalized.append(f"{sha} {msg}".strip())
            self.commits = normalized

    def full_text(self) -> str:
        parts = [self.subject, self.body]

        if self.pr_numbers:
            parts.append("PR Numbers:\n" + ", ".join(self.pr_numbers))

        if self.repos:
            parts.append("Repos:\n" + ", ".join(self.repos))

        if self.tickets:
            parts.append("Tickets:\n" + ", ".join(self.tickets))

        if self.pr_title:
            parts.append("PR Title:\n" + self.pr_title)

        if self.commits:
            parts.append("Commits:\n" + "\n".join(self.commits))

        if self.files_modified:
            parts.append("Files Modified:\n" + "\n".join(self.files_modified))

        if self.change_counts:
            parts.append("Change Counts:\n" + str(self.change_counts))
        
        if self.body:
            parts.append("Body:\n" + self.body[:2000])  # Limit body to first 2000 chars

        return "\n\n".join(parts)
