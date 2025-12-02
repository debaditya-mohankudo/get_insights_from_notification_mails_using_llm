from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EmailMessage(BaseModel):
    """
    Strictly validated Pydantic v2 email object for your GitHub Notification Index Engine.
    """

    # --------------------------
    # Basic metadata
    # --------------------------
    subject: str
    sender: Optional[str] = None
    date: Optional[str] = None
    message_id: Optional[str] = None

    # --------------------------
    # PR-related metadata
    # --------------------------
    pr_numbers: Optional[List[int]] = None
    pr_title: Optional[str] = None
    repos: Optional[List[str]] = None
    tickets: Optional[List[str]] = None

    # --------------------------
    # Email text + parsed content
    # --------------------------
    body: str
    markdown: Optional[dict] = None
    commits: Optional[List[str]] = None
    files_modified: Optional[List[str]] = None

    # --------------------------
    # Additional attributes
    # --------------------------
    tags: Optional[List[str]] = None

    # ============================================================
    # FIELD-LEVEL VALIDATORS (v2)
    # ============================================================

    @field_validator("pr_numbers", mode="before")
    def convert_pr_numbers(cls, value):
        """
        Convert PR numbers from strings → ints.
        Accepts:
            ["8040"] → [8040]
            8040 → [8040]
            None → None
        """
        if not value:
            return None

        if isinstance(value, int):
            return [value]

        return [int(v) for v in value]

    @field_validator(
        "repos",
        "tickets",
        "markdown",
        "commits",
        "files_modified",
        "tags",
        mode="before"
    )
    def empty_list_to_none(cls, value):
        """Convert [] → None."""
        if value is None or value == []:
            return None
        return value

    # ============================================================
    # MODEL-LEVEL VALIDATORS (v2)
    # ============================================================

    @model_validator(mode="after")
    def clean_empty_fields(self):
        """
        Normalize fields after object creation.
        - Turn empty strings into None
        - Normalize PR title spacing
        """
        if isinstance(self.pr_title, str) and self.pr_title.strip() == "":
            self.pr_title = None
        return self

    # ============================================================
    # EMBEDDING TEXT BUILDER
    # ============================================================

    def full_text(self) -> str:
        """
        Combined text for embeddings:
        Includes PR title, repos, PR numbers, tickets, markdown, commits,
        files modified, and the raw body.
        """
        parts = []

        if self.pr_title:
            parts.append(f"Title: {self.pr_title}")

        if self.pr_numbers:
            parts.append(f"PR Numbers: {', '.join(map(str, self.pr_numbers))}")

        if self.repos:
            parts.append(f"Repos: {', '.join(self.repos)}")

        if self.tickets:
            parts.append(f"Tickets: {', '.join(self.tickets)}")

        if self.markdown:
            md_parts = []
            for section, items in self.markdown.items():
                if items:
                    md_parts.append(f"## {section}")
                    md_parts.extend(f"- {i}" for i in items)
            if md_parts:
                parts.append("Markdown Sections:\n" + "\n".join(md_parts))


        if self.commits:
            parts.append("\n".join(self.commits))

        if self.files_modified:
            parts.append("\n".join(self.files_modified))
        
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        parts.append(self.body)

        return "\n\n".join(parts)
