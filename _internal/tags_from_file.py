# tag_from_files.py
"""
Extract semantic tags based on file paths modified in the PR.
"""

from typing import List, Set


FILE_RULES = {
    "ui": [
        "/ui/", "/frontend/", "/components/", "/views/", "/templates/",
        ".css", ".scss", ".sass", ".less",
        ".jsx", ".tsx", ".vue", ".html", ".phtml"
    ],
    "sql": [
        "/migrations/", "/migration/", "/db/", "/database/",
        ".sql"
    ],
    "api": [
        "/api/", "/routes/", "/controllers/", "/endpoints/",
        "router", "controller"
    ],
    "security": [
        "/auth/", "/authentication/", "/authorization/",
        "jwt", "oauth", "/security/", "permissions"
    ],
    "performance": [
        "cache", "caching", "/utils/perf", "/performance/",
        "indexing", "batch", "async", "concurrency"
    ],
    "backend": [
        "/services/", "/models/", "/handlers/", "/core/",
        ".go", ".rb", ".py", ".java", ".ts", ".php"
    ],
}


def classify_tags_from_files(files: List[str]) -> List[str]:
    """
    Classify PR tags based purely on files modified.
    """
    tags: Set[str] = set()

    for file in files:
        f = file.lower()
        for tag, patterns in FILE_RULES.items():
            for p in patterns:
                if p in f:
                    tags.add(tag)
                    break

    return sorted(tags)