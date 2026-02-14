import os
import pickle
from pathlib import Path
from typing import List
import multiprocessing

import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count

from _internal.email_models import EmailMessage
from _internal.extract_emails_from_mbox import EmailExtractor

# ============================================================
#                   INDEX STORAGE SETUP
# ============================================================      
INDEX_DIR = Path("./index_data")
os.makedirs(INDEX_DIR, exist_ok=True)
print(f"✔ Index directory ready at: {INDEX_DIR.resolve()}")


# ============================================================
#               WORKER FUNCTION (MULTIPROCESS SAFE)
# ============================================================

def process_single_mbox(path: Path) -> List[EmailMessage]:
    """
    Extracts emails from a single mbox file.
    Runs inside a separate worker process (fork safe on macOS).
    """
    extractor = EmailExtractor()
    return extractor.extract_emails_from_mbox(path)


# ============================================================
#               EMBEDDING + INDEX CREATION
# ============================================================

def embed_and_index(
    emails: List[EmailMessage],

    index_path=os.path.join(INDEX_DIR, "index.faiss"),
    meta_path=os.path.join(INDEX_DIR, "meta.pkl")
    
):
    print("🔢 Generating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [email.full_text() for email in emails]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]

    print("📦 Creating FAISS index...")
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efSearch = 64
    index.hnsw.efConstruction = 200

    index.add(embeddings) # type: ignore

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(emails, f)

    print(f"✔ FAISS index saved to: {index_path}")
    print(f"✔ Metadata saved to: {meta_path}")


# ============================================================
#               MAIN EXECUTION (PARALLEL + FORK MODE)
# ============================================================

if __name__ == "__main__":
    # Fix macOS semaphore leak warnings
    multiprocessing.set_start_method("fork")

    ROOT = Path("/Users/debaditya/workspace/trash_mails")

    # Collect all .mbox/mbox files
    mbox_files = []
    for item in ROOT.iterdir():
        if item.suffix == ".mbox":
            mbox_path = os.path.join(item, "mbox")
            if os.path.exists(mbox_path):
                mbox_files.append(str(mbox_path))

    print(f"📦 Total mbox files discovered: {len(mbox_files)}")

    # --------------------------------------------------------
    # PARALLEL EXTRACTION WITH ProcessPoolExecutor
    # --------------------------------------------------------
    print("⚡ Extracting emails in parallel...")

    emails: List[EmailMessage] = []

    with ThreadPoolExecutor(max_workers=cpu_count()//2) as executor:
        futures = {executor.submit(process_single_mbox, path): path for path in mbox_files}

        for future in tqdm(as_completed(futures), total=len(futures)):
            batch = future.result()
            emails.extend(batch)

    print(f"📨 Total emails extracted: {len(emails)}")

    # --------------------------------------------------------
    # EMBEDDINGS + INDEX
    # --------------------------------------------------------
    embed_and_index(emails)