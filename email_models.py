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

    commits: Optional[List[str]] = None
    files_modified: Optional[List[str]] = None
    change_counts: Optional[Dict[str, int]] = None

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

        return "\n\n".join(parts)
