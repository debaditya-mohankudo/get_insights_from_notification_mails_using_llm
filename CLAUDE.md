# CLAUDE.md

## Project Overview

Offline RAG (Retrieval-Augmented Generation) system for querying GitHub notification emails. Parses Apple Mail `.mbox` exports, indexes them with FAISS, and answers natural-language queries using a local LLM (Ollama).

## Entry Points

- `build_index.py` — builds FAISS index + metadata pickle from mbox files
- `query_llm.py` — queries the index; usage: `python query_llm.py "your question"`
- `convert_mbox_maildir.py` — utility to convert mbox → maildir format

## Running the Project

**Prerequisites:** Python 3.10+, Ollama running locally with `llama3.2:3b` pulled.

```bash
# Install dependencies
pip install -r requirements.txt

# Build index (reads mbox files from /Users/debaditya/workspace/trash_mails/)
python build_index.py

# Query
python query_llm.py "what PRs touched the auth module?"
```

## Project Structure

```
_internal/
  email_models.py          # Pydantic v2 EmailMessage (core data model)
  extract_emails_from_mbox.py  # EmailExtractor class
  helpers.py               # Regex-based metadata extraction
  tag_classifier.py        # MiniLM semantic tag classification (30+ tags)
  tags_from_file.py        # Rule-based file-path → tag mapping
  markdown_sections.py     # Extracts code blocks, headings, lists from email body
index_data/                # Generated artifacts (gitignored)
  index.faiss              # FAISS HNSW vector index
  meta.pkl                 # Pickled list of EmailMessage objects
```

## Architecture

**Pipeline:** `.mbox` files → extract → validate (Pydantic) → tag → embed (MiniLM) → FAISS index

**Query dispatch (3 modes):**
1. **Commit mode** — query contains a 7–40 char hex string → filters by commit SHA
2. **PR mode** — query contains `PR #N` → filters by PR number
3. **Semantic mode** — hybrid: vector similarity + exact-match scoring + tag overlap reranking → Ollama synthesis

**Merging:** Multiple emails about the same PR are merged into a single `EmailMessage` via `append_by_pr()`.

## Core Classes & Functions

| Symbol | File | Purpose |
|--------|------|---------|
| `EmailMessage` | `_internal/email_models.py` | Pydantic model; holds all PR metadata |
| `EmailExtractor` | `_internal/extract_emails_from_mbox.py` | Parses mbox, runs parallel extraction |
| `extract_metadata_from_subject()` | `_internal/helpers.py` | Regex extracts PR#, repo, commit SHAs |
| `extract_tags_miniLM()` | `_internal/tag_classifier.py` | Semantic tag classification |
| `tags_from_file_paths()` | `_internal/tags_from_file.py` | Rule-based file-path tagging |
| `score_email()` | `query_llm.py` | Hybrid scorer: field weights + tag overlap |
| `build_context()` | `query_llm.py` | Assembles LLM prompt context from top results |

## Key Design Decisions

- **Tags from 4 sources:** title, commits, files, markdown headings — unioned into `EmailMessage.tags`
- **File paths split into components** (`src/utils/helpers.js` → `["src", "utils", "helpers.js"]`) for finer tag matching
- **FAISS HNSW** (not flat search): 32 neighbors, efSearch=64, efConstruction=200
- **Batch embedding size:** 32; parallel extraction uses `cpu_count()//2` workers
- **Pydantic validators** normalize types (empty lists → None, PR numbers as ints)

## Hardcoded Paths (to change if moving environments)

- mbox source root: `/Users/debaditya/workspace/trash_mails/` (in `build_index.py`)
- Index output/input: `./index_data/` (relative to repo root)

## External Dependency

Ollama must be running: `ollama serve` with `llama3.2:3b` pulled (`ollama pull llama3.2:3b`).

## Testing

```bash
pytest tests/
```

Tests are minimal. The `index_data/` directory and `__pycache__/` are gitignored.
