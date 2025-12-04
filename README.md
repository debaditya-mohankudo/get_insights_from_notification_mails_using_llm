(Auto-generated)


---

# 📘 **README.md — GitHub Notification Email Insight Engine**

This project transforms your exported GitHub notification emails into a **searchable vector database**, enabling powerful natural-language queries about pull requests using a **local LLM** (Ollama).
It is fully offline, fast, and optimized for PR-specific retrieval.

---

# 🚀 Features

* Parse Apple Mail GitHub notification `.mbox` files
* Extract structured metadata:

  * PR numbers
  * Repository names
  * Jira tickets
  * Commit SHAs (normalized to first 7 chars)
  * Files modified
  * PR titles
  * Markdown sections (headings, lists, code blocks)
* Build a **FAISS HNSW** vector index
* Hybrid retrieval:

  * Weighted exact match
  * Semantic search fallback
  * Strict PR filter
* Query using a local model (`llama3.2:3b`)
* Returns commit summaries, file changes, PR description, etc.

---

# 🧱 Project Structure

```
.
├── build_index.py           # Parse emails, extract metadata, build FAISS index
├── query_llm.py             # Query engine + LLM orchestration
├── email_models.py          # EmailMessage dataclass (commit truncation, full text)
├── markdown_sections.py     # Markdown extraction utilities
├── index.faiss              # Generated vector index
├── meta.pkl                 # Serialized EmailMessage objects
└── *.mbox/mbox              # Raw exported GitHub Apple Mail folders
```

---

# 📥 1. Preparing Your Data

Export your GitHub notification folders from **Apple Mail**:

1. Select the mailbox
2. Right click → **Export Mailbox**
3. Place all exported folders in your project directory
4. Each exported folder contains a `mbox` file:

   ```
   repo_notifications.mbox/mbox
   ```

---

# 🏗 2. Building the Index

Run:

```bash
python build_index.py
```

What this script does:

✔ Loads all `*.mbox/mbox` files
✔ Parses each email body (plain + HTML → cleaned)
✔ Extracts:

* PR numbers (subject + Message-ID)
* Repos
* Tickets
* Clean PR title
* Commit list (regex match, normalized to 7 chars)
* Files modified
* Markdown sections
  ✔ Builds a combined text representation via `EmailMessage.full_text()`
  ✔ Encodes using **SentenceTransformers: all-MiniLM-L6-v2**
  ✔ Saves:

```
index.faiss
meta.pkl
```

---

# 🔍 3. Querying a PR

Use natural language:

```bash
python query_llm.py "pr #1234 commits and file changes"
```

The query engine:

### Step 1 — Extract PR number

Example: `"1234"`

### Step 2 — Weighted exact match

Scores PR numbers, repos, tickets, commits, file paths, and PR title.

### Step 3 — Strict PR filter

Ensures **only emails belonging to PR 1234** are considered.

### Step 4 — Semantic search fallback

Augments the query with all repo/PR/ticket tokens to improve vector recall.

### Step 5 — Format context chunks

Includes:

* commits
* files modified
* markdown code blocks
* headings
* lists
* first 1500 chars of email body

### Step 6 — Local LLM processing

Uses:

```python
ollama.generate(model="llama3.2:3b")
```

The LLM is instructed to:

* Answer **only about the exact PR**
* Use only retrieved email fragments
* Avoid hallucination

---

# 🧠 Example Output

```
[Exact-match retrieval → 5 results]

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

# 🧩 Why This Works So Well

This project achieves high-precision PR answers because it uses:

### ✔ **Hybrid search**

Exact + semantic retrieval
→ almost never returns wrong PR emails.

### ✔ **Metadata-rich indexing**

Commits, PR titles, repos, tickets are treated as first-class search features.

### ✔ **Context shaping**

Context blocks contain structured + raw content.

### ✔ **Local LLM with clear rules**

Reduces hallucination significantly.

---

# 🔧 Requirements

* Python 3.10+
* FAISS
* sentence-transformers
* BeautifulSoup4
* tqdm
* Ollama (for local LLM)

Install:

```bash
pip install faiss-cpu sentence-transformers beautifulsoup4 tqdm
```

You must also have:

```bash
brew install ollama
ollama pull llama3.2:3b
```

---

# 🧪 Optional Improvements

I can help you implement:

* Change counting per file (`+/-` lines)
* Repo-level analytics
* Query augmentation via RAG-chain
* Web UI
* Embedding optimization (bge-small-en, nomic-embed-text, etc.)
* PR graph linking (threads/comments/commits)

---

# 🎉 You're Done!

You now have a fully local, private, intelligent PR knowledge engine powered by GitHub emails.

---

If you'd like, I can also generate:

✅ A **diagram** of the data flow
✅ A **project architecture SVG**
✅ A **flowchart**
✅ A **demo GIF**
Just tell me which one.

