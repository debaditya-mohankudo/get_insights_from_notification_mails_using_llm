import re
from typing import List, Dict, Optional


# ============================================================
#                 REGEX DEFINITIONS
# ============================================================

FENCED_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_\-]*)\n(.*?)```",
    re.DOTALL
)

HEADING_RE = re.compile(
    r"^(#{1,6})\s+(.*)$",
    re.MULTILINE
)

LIST_RE = re.compile(
    r"^[ \t]*([-*+]|\d+\.)\s+.+$",
    re.MULTILINE
)


# ============================================================
#                 EXTRACTION HELPERS
# ============================================================

def extract_code_blocks(text: str) -> Optional[List[str]]:
    """
    Returns list of fenced code blocks content.
    """
    blocks = FENCED_BLOCK_RE.findall(text)
    return [b.strip() for b in blocks] or None


def extract_headings(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Returns headings with their levels.
    Example: { "level": 2, "title": "Installation" }
    """
    matches = HEADING_RE.findall(text)
    if not matches:
        return None

    return [
        {"level": str(len(h[0])), "title": h[1].strip()}
        for h in matches
    ]


def extract_lists(text: str) -> Optional[List[str]]:
    """
    Extract bullet or numbered list lines from markdown.
    """
    lines = LIST_RE.findall(text)
    
    if not lines:
        return None

    # Re-run regex to extract full lines (not just bullet symbol)
    raw_lines = re.findall(LIST_RE.pattern, text, re.MULTILINE)
    cleaned = [l.strip() for l in raw_lines]

    return cleaned or None


# ============================================================
#        HIGH LEVEL FUNCTION FOR BUILD_INDEX INTEGRATION
# ============================================================

def extract_markdown_sections(text: str) -> Dict[str, Optional[List]]:
    """
    Extracts markdown features:
    - code blocks
    - headings
    - bullet lists

    Can be safely added to EmailMessage.
    """
    return {
        "code_blocks": extract_code_blocks(text),
        "headings": extract_headings(text),
        "lists": extract_lists(text),
    }