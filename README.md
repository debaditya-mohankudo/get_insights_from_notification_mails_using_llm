# GitHub Notification Email Insight Engine

A **fully local, offline RAG system** that parses exported Apple Mail GitHub notification `.mbox` files and enables natural-language querying about PRs, commits, and code changes using a **local LLM** (Ollama).

---

## Architecture

```
.mbox files --> build_index.py --> FAISS index + metadata pickle
                                          |
                              query_llm.py --> Ollama (llama3.2:3b) --> answer
```

---

## Key Components

| File | Purpose |
|---|---|
| `build_index.py` | Parallel mbox extraction, SentenceTransformer embeddings, FAISS HNSW index |
| `query_llm.py` | 3 query modes: **commit**, **PR**, **semantic**. Hybrid retrieval (exact match + vector search) with Ollama LLM |
| `_internal/email_models.py` | Pydantic v2 `EmailMessage` model with validators and `append_by_pr()` merge strategy |
| `_internal/extract_emails_from_mbox.py` | `EmailExtractor` - parses mbox, extracts metadata, classifies tags, merges by PR |
| `_internal/helpers.py` | Regex-based extraction: PR numbers, repos, tickets, commits, files, contributors |
| `_internal/tag_classifier.py` | Semantic tag classification via MiniLM cosine similarity against a predefined tag set |
| `_internal/tags_from_file.py` | Rule-based tag classification from file paths |
| `_internal/markdown_sections.py` | Extracts code blocks, headings, and lists from email bodies |
| `convert_mbox_maildir.py` | Utility to convert mbox to maildir format |

---

## Features

* Parse Apple Mail GitHub notification `.mbox` files
* Extract structured metadata:
  * PR numbers, repository names, Jira tickets
  * Commit SHAs (normalized to first 7 chars)
  * Files modified (split into path components for finer tagging)
  * PR titles, contributors
  * Markdown sections (headings, lists, code blocks)
* Emails **merged by PR number** (`append_by_pr`) so each PR becomes a single rich document
* **Semantic tag classification** using MiniLM embedding similarity against 30 predefined tags
* Tags derived from 3 sources: PR title, commit messages, and file paths
* Build a **FAISS HNSW** vector index
* Hybrid retrieval:
  * Weighted exact match scoring
  * Semantic vector search
  * Tag-based reranking
  * Strict PR filter
* Query using a local model (`llama3.2:3b`)

---

## Tech Stack

* **FAISS** (HNSW) for vector indexing
* **SentenceTransformers** (`all-MiniLM-L6-v2`) for embeddings + tag classification
* **Ollama** (`llama3.2:3b`) for answer generation
* **Pydantic v2** for data modeling
* **BeautifulSoup4** for HTML parsing
* **tqdm** for progress bars
* Python 3.10+

---

## Preparing Your Data

Export your GitHub notification folders from **Apple Mail**:

1. Select the mailbox
2. Right click -> **Export Mailbox**
3. Place all exported folders in your project directory
4. Each exported folder contains a `mbox` file:

   ```
   repo_notifications.mbox/mbox
   ```

---

## Building the Index

```bash
python build_index.py
```

What this script does:

1. Discovers all `*.mbox/mbox` files
2. Extracts emails in parallel using `ThreadPoolExecutor`
3. Parses each email body (plain + HTML, cleaned via BeautifulSoup)
4. Extracts PR numbers (subject + Message-ID), repos, tickets, clean PR title, commits, files modified, markdown sections, contributors
5. Classifies tags using MiniLM similarity (title, commits, file paths, section headings)
6. Merges emails by PR number into single rich documents
7. Encodes using **SentenceTransformers: all-MiniLM-L6-v2**
8. Saves `index.faiss` and `meta.pkl` to `index_data/`

---

## Querying

```bash
python query_llm.py "pr #1234 commits and file changes"
```

The query engine operates in 3 modes:

### Commit Mode

Activated when the query contains a commit hash. Filters emails by commit and summarizes.

### PR Mode

Activated when the query contains `PR #N` or similar patterns. Filters emails by PR number and summarizes using Ollama.

### Semantic Mode

Fallback when no PR/commit is detected:

1. Extracts tags from the query using MiniLM + file-path rules
2. Augments query with extracted tags
3. Performs FAISS vector search (top 5)
4. Reranks results by exact-match scoring (PR numbers, repos, tickets, commits, files, tags, contributors)
5. Reranks again by tag overlap
6. Builds context and generates answer via Ollama

---

## Example Output

```
[Exact-match retrieval -> 5 results]

==================================================
PR 1234 Summary:
- 5 commits
- 3 files modified
- Fixes on purchase history UI
- Replaced HTML download with JS PDF download
...
==================================================
```

---

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `faiss-cpu`, `numpy`, `tqdm`, `sentence-transformers`, `transformers`, `torch`, `protobuf`, `ollama`

Also requires [Ollama](https://ollama.ai) running locally with the `llama3.2:3b` model pulled.
