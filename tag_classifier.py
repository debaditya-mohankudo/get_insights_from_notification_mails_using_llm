# tag_classifier.py
"""
Tag classifier for PR titles using:
1. Keyword-based semantic rules
2. (Optional) Embedding-based similarity scoring

Used across the indexing pipeline to attach tags to EmailMessage objects.
"""

from typing import List, Set
import re

# ------------------------------------------------------
# 1. SEMANTIC KEYWORD RULES
# ------------------------------------------------------

RULES = {
    "bug": [
        r"\bbug\b",
        r"\bfix\b",
        r"\berror\b",
        r"\bissue\b",
        r"\bcrash\b",
        r"\bhotfix\b",
    ],
    "sql": [
        r"\bsql\b",
        r"\btable\b",
        r"\bdatabase\b",
        r"\bdb\b",
        r"\bquery\b",
    ],
    "ui": [
        r"\bui\b",
        r"\bux\b",
        r"\bfrontend\b",
        r"\bbutton\b",
        r"\blayout\b",
        r"\bdesign\b",
    ],
    "api": [
        r"\bapi\b",
        r"\bendpoints?\b",
        r"\brest\b",
        r"\bjson\b",
    ],
    "security": [
        r"\bsecurity\b",
        r"\bxss\b",
        r"\bsql[\s_-]?injection\b",
        r"\bauth(entication|orization)?\b",
        r"\bcsrf\b",
    ],
    "performance": [
        r"\bperformance\b",
        r"\bspeed\b",
        r"\bfaster\b",
        r"\boptimi[sz](e|ing|ation)?\b",
        r"\blatency\b",
    ],
}

# ------------------------------------------------------
# 2. MAIN TAG GENERATION FUNCTION (Semantic Rules Only)
# ------------------------------------------------------

def generate_tags_from_pr_title(pr_title: str) -> List[str]:
    """
    Generate tags for a PR title using simple semantic keyword matching.
    This is fast and runs during indexing inside build_index.py.
    """
    if not pr_title:
        return []

    title = pr_title.lower()
    tags: Set[str] = set()

    for tag, patterns in RULES.items():
        for pattern in patterns:
            if re.search(pattern, title):
                tags.add(tag)
                break

    return sorted(tags)


# ------------------------------------------------------
# 3. OPTIONAL: EMBEDDING-BASED TAGGING (Plug-in API)
# ------------------------------------------------------

"""
You may optionally use embeddings later.

Example:

def generate_embedding_tags(pr_title, embedding_model):
    emb = embedding_model.encode([pr_title])[0]
    scores = cosine similarity with predefined tag vectors
    return tags above threshold

Not implemented now because keyword rules already work well.
"""


# ------------------------------------------------------
# 4. COMBINED TAGGING PIPELINE
# ------------------------------------------------------

def classify_tags(pr_title: str) -> List[str]:
    """
    Unified tagging entrypoint. Currently only semantic rules.
    Later you can combine semantic + embedding scores here.
    """
    tags = set()

    # semantic rules
    tags.update(generate_tags_from_pr_title(pr_title))

    return sorted(tags)
