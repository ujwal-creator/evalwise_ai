
import hashlib
import os
import re

import chromadb


DB_DIR = os.getenv("CHROMA_DB_DIR", "chroma_db")
COLLECTION_PREFIX = "evalwise_"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def _collection_name(namespace):
    """Create a valid, stable ChromaDB collection name."""
    raw = str(namespace or "default")
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]
    return f"{COLLECTION_PREFIX}{digest}"


def _client():
    """Open the persistent local ChromaDB database."""
    os.makedirs(DB_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=DB_DIR)


def _chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long reference documents into overlapping text chunks."""
    text = str(text or "").strip()
    if not text:
        return []

    if overlap >= size:
        overlap = size // 5

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def build_index(documents, namespace="default"):
    """
    Store reference documents in a persistent ChromaDB collection.

    documents: list of dictionaries with source, kind, and text fields.
    namespace: separates the references for each evaluation.
    """
    client = _client()
    name = _collection_name(namespace)

    # Rebuild this evaluation's collection to avoid duplicate/stale passages.
    try:
        client.delete_collection(name)
    except Exception:
        pass

    collection = client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    texts = []
    metadatas = []

    for doc_index, item in enumerate(documents or []):
        if not isinstance(item, dict):
            continue

        source = str(item.get("source") or "Unknown source")
        kind = str(item.get("kind") or "reference")
        text = str(item.get("text") or "")

        for chunk_index, chunk in enumerate(_chunks(text)):
            ids.append(f"doc{doc_index}_chunk{chunk_index}")
            texts.append(chunk)
            metadatas.append({
                "source": source[:500],
                "kind": kind[:100],
                "chunk_index": chunk_index,
            })

    if texts:
        # Batch inserts to avoid oversized requests.
        batch_size = 100
        for start in range(0, len(texts), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
            )

    return len(texts)


def retrieve_context(query, namespace="default", top_k=8):
    """
    Retrieve the most relevant reference passages.

    Returns a list of dictionaries compatible with evaluator.py.
    """
    query = str(query or "").strip()
    if not query:
        return []

    client = _client()
    name = _collection_name(namespace)

    try:
        collection = client.get_collection(name=name)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(max(1, int(top_k)), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    passages = []

    for index, text in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        passages.append({
            "source": metadata.get("source", "Unknown source"),
            "kind": metadata.get("kind", "reference"),
            "text": text,
            "distance": distance,
        })

    return passages
