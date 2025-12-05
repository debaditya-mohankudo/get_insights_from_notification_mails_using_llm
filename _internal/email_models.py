from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from _internal.helpers import CommitInfo


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
    commits: Optional[List[CommitInfo]] = None
    files_modified: Optional[List[str]] = None

    # --------------------------
    # Additional attributes
    # --------------------------
    tags: Optional[List[str]] = None
    linked_prs: Optional[List[int]] = None
    linked_tickets: Optional[List[str]] = None
    contributors: Optional[List[str]] = None

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
        "contributors",
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
    
    def append_by_pr(self, result=[]) -> List:
        """
        results: List[EmailMessage]
        email_msg: EmailMessage
        get pr_number from email_msg
        check if result contains email with same pr_number
        if yes, merge the two email messages( merge all the fields excpet body and sender and date, any field that is not a list, if existing field is None, update it with new field)
        markdown is merged by appending the lists inside dictionary
        skip sonar related markdown merging
        body is merged by appending the bodies with two new lines
        if no, append email_msg to results
        """
        pr_numbers = self.pr_numbers or []
        if result == []:
            result.append(self)
            return result

        for pr in pr_numbers:
            found = False
            for existing_email in result:
                if existing_email.pr_numbers and pr in existing_email.pr_numbers:
                    # Merge fields
                    for field in self.model_fields_set - {'sender', 'date', 'subject', 'message_id', 'pr_title', 'pr_numbers'}:
                        existing_value = getattr(existing_email, field)
                        new_value = getattr(self, field)

                        if isinstance(existing_value, list):
                            if new_value:
                                combined = set(existing_value) | set(new_value)
                                setattr(existing_email, field, list(combined))
                        else:
                            if existing_value is None and new_value is not None:
                                setattr(existing_email, field, new_value)
                            elif field == "markdown":
                                if existing_value and new_value:
                                    for key, items in new_value.items():
                                        if key in existing_value and items:
                                            if existing_value[key]:
                                                if 'sonar' not in key.lower() or 'sonar' not in ",".join(items).lower():
                                                    existing_value[key].extend(items)
                                        else:
                                            existing_value[key] = items
                                    setattr(existing_email, field, existing_value)
                            elif field == "body":
                                # Append bodies
                                if new_value:
                                    combined_body = "\n\n".join([existing_value, new_value])
                                    setattr(existing_email, field, combined_body)
                    found = True
                    break
            if not found:
                result.append(self)
        return result
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

        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")

        if self.pr_title:
            parts.append(f"Title: {self.pr_title}")

        if self.pr_numbers:
            parts.append(f"PR Numbers: {', '.join(map(str, list(set(self.pr_numbers))))}")

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
            parts.append("\n".join(f"{short},{message}" for sha, short, message in self.commits))

        if self.files_modified:
            parts.append("\n".join(self.files_modified))

        if self.linked_prs:
            parts.append(f"Linked PRs: {', '.join(map(str, self.linked_prs))}")

        if self.linked_tickets:
            parts.append(f"Linked Tickets: {', '.join(self.linked_tickets)}")

        if self.contributors:
            parts.append(f"Contributors: {', '.join(self.contributors)}")

        parts.append(self.body)

        return "\n\n".join(parts)
