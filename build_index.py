import os
import re
import mailbox
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup

from email_models import EmailMessage
from extract_emails_from_mbox import EmailExtractor
from helpers import clean_html, extract_metadata_from_subject, extract_commits, extract_files_modified
from markdown_sections import extract_markdown_sections


# ============================================================
#               INDEX BUILDER
# ============================================================

def embed_and_index(emails: List[EmailMessage], index_path="index.faiss", meta_path="meta.pkl"):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [email.full_text() for email in emails]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    dim = embeddings.shape[1]

    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200

    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(emails, f)

    print(f"✔ FAISS index saved: {index_path}")
    print(f"✔ Metadata saved: {meta_path}")


# ============================================================
#               MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    ROOT = Path("/Users/debaditya/workspace/trash_mails")

    emails: List[EmailMessage] = []

    for item in ROOT.iterdir():
        # Only parse folder/*.mbox/mbox
        if item.suffix == ".mbox":
            mbox_path = item / "mbox"
            if mbox_path.exists():
                extractor = EmailExtractor()
                emails.extend(extractor.extract_emails_from_mbox(str(mbox_path)))


    print(f"📨 Total emails collected: {len(emails)}")

    embed_and_index(emails)