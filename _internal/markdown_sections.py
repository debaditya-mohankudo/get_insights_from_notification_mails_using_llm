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

# NEW: Support GitHub email plain-text headings
PLAIN_HEADING_RE = re.compile(
    r"^(Commit Summary|File Changes|What changed\?|What changed|Summary|"
    r"Implementation Details|Implementation|Testing Notes|Changelog|Description)\s*(?:\(.+\))?$",
    re.IGNORECASE | re.MULTILINE
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


def extract_heading_sections_with_content(text: str):
    """
    Extracts sections of the form:
        [
            ("Heading 1", [line1, line2, ...]),
            ("Heading 2", [...]),
            (None, [...])   # text before first heading
        ]

    Uses both markdown (# ## ###) and plain GitHub headings
    that your enhanced PLAIN_HEADING_RE detects.
    """

    sections = []
    current_heading = None
    current_lines = []

    # iterate line-by-line
    for line in text.splitlines():
        raw = line.rstrip()
        if not raw.strip():
            continue

        # markdown-style heading
        m = HEADING_RE.match(raw)
        if m:
            # save previous
            if current_heading is not None or current_lines:
                sections.append((current_heading, current_lines))
            current_heading = m.group(2).strip()
            current_lines = []
            continue

        # plain-text GitHub heading
        m2 = PLAIN_HEADING_RE.match(raw)
        if m2:
            if current_heading is not None or current_lines:
                sections.append((current_heading, current_lines))
            current_heading = m2.group(1).strip()
            current_lines = []
            continue

        # normal content line
        current_lines.append(raw)

    # add last section
    if current_heading is not None or current_lines:
        sections.append((current_heading, current_lines))

    return sections


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
        "headings": extract_heading_sections_with_content(text),
        "lists": extract_lists(text),
    }

