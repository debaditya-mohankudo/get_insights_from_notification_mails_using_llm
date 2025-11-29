import sys, os, json, pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2:3b"
HISTORY_FILE = "conversation_history.json"


# ---------------------------
# HISTORY MANAGEMENT
# ---------------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    return json.load(open(HISTORY_FILE))

def save_history(history):
    json.dump(history, open(HISTORY_FILE, "w"), indent=2)


# ---------------------------
# FAISS SEARCH
# ---------------------------

def search_faiss(query, k=5):
    index = faiss.read_index("index.faiss")

    with open("meta.pkl", "rb") as f:
        meta = pickle.load(f)

    model = SentenceTransformer(EMBED_MODEL)
    q_emb = model.encode([query]).astype("float32")

    D, I = index.search(q_emb, k)
    return [meta[i] for i in I[0]]


def build_email_context(results):
    ctx = ""
    for r in results:
        ctx += f"\nSubject: {r['subject']}\n"
        ctx += f"From: {r['from']}\n"
        ctx += f"Date: {r['date']}\n"
        ctx += r['body'][:3000]
        ctx += "\n---\n"
    return ctx


# ---------------------------
# LLM CALL
# ---------------------------

def ask_llm(query, email_context, history):
    history_text = ""
    for turn in history:
        history_text += f"{turn['role'].upper()}: {turn['content']}\n"

    prompt = f"""
You are an email-based RAG assistant.
Below is the conversation so far:

{history_text}

User's new question:
{query}

Relevant emails from search:
{email_context}

Give a helpful, concise, context-aware answer.
"""

    response = ollama.generate(model=LLM_MODEL, prompt=prompt)
    return response["response"]


# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python query_llm_memory.py \"your question\"")
        sys.exit(1)

    query = sys.argv[1]

    # Load conversation history
    history = load_history()

    # RAG retrieval
    results = search_faiss(query, k=5)
    email_context = build_email_context(results)

    # LLM answer
    answer = ask_llm(query, email_context, history)

    # Print result nicely
    print("\n" + "=" * 50)
    print(answer)
    print("=" * 50 + "\n")

    # Add to history
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    save_history(history)

