# tag_classifier.py
"""
Tag classifier using: LLM-based classification.
The LLM is allowed to choose from a predefined set of tags only.

Used across the indexing pipeline to attach tags to EmailMessage objects.
"""

from typing import List, Set
import re, json

import ollama
from sentence_transformers import SentenceTransformer, util

ALLOWED_TAGS = [
    "bug", 
    "fix", 
    "feature",
    "refactor", 
    "docs", 
    "performance",
    "security", 
    "breaking_change", 
    "test", 
    "dependency", 
    "ui", 
    "api",
    "backend", 
    "authentication", 
    "sql",
    "middleware", 
    "logging", 
    "monitoring",
    "exceptions_handling",
    "billing",
    "subscriptions",
    "notifications",
    "search",
    "caching",
    "load_balancing",
    "docker",
    "cron_jobs",
    "data_migration",
    "cryptography",
    "password_management",
    "order_lifecycle",
    "shopping_cart",
    "payment_processing",

    # add more as needed
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def extract_tags_miniLM(text="", top_k: int = 5) -> list[str]:
    query_emb = model.encode(text, convert_to_tensor=True)
    tag_embs = model.encode(ALLOWED_TAGS, convert_to_tensor=True)

    sims = util.cos_sim(query_emb, tag_embs)[0]
    top_idx = sims.topk(top_k).indices.tolist()

    return [ALLOWED_TAGS[i] for i in top_idx]