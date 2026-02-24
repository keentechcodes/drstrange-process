#!/usr/bin/env python3
"""ingest_milvus.py — Embed textbook chunks and insert into Zilliz Cloud (Milvus).

Pipeline step 5 (final):
    chunk_textbook.py → BATES.chunks.json → [this script] → Zilliz Cloud collection

This script:
  1. Connects to Zilliz Cloud using credentials from .env / environment
  2. Creates the collection with schema + BM25 function (if not exists)
  3. Loads chunks from the JSON produced by chunk_textbook.py
  4. Generates 768-dim dense embeddings with S-PubMedBert-MS-MARCO
  5. Inserts chunks in batches (sparse vectors auto-generated server-side by BM25)

Environment variables (or .env file):
    ZILLIZ_ENDPOINT  — Public endpoint URL
    ZILLIZ_TOKEN     — Cluster token (user:pass) or API key

Usage:
    # Full ingestion
    python ingest_milvus.py results_book1/BATES.chunks.json

    # Dry run — embed only, no Milvus connection (validates data + model)
    python ingest_milvus.py results_book1/BATES.chunks.json --dry-run

    # Re-create collection (drops existing data!)
    python ingest_milvus.py results_book1/BATES.chunks.json --recreate

    # Custom batch size
    python ingest_milvus.py results_book1/BATES.chunks.json --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Embedding model — must match the tokenizer in chunk_textbook.py
# S-PubMedBert-MS-MARCO: PubMedBERT fine-tuned on MS-MARCO for retrieval.
# 768-dim, mean pooling, max_seq_length=350.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_DIM = 768

# Milvus collection name
COLLECTION_NAME = "textbook_chunks"

# Insertion batch size (Zilliz free tier handles up to ~100 per insert)
DEFAULT_BATCH_SIZE = 64

# Maximum VARCHAR length for the text field (BM25-enabled).
# Milvus VARCHAR max is 65535 bytes.  Our largest chunk is ~1700 tokens
# which is well within this limit.
TEXT_MAX_LENGTH = 65535


# ── helpers ────────────────────────────────────────────────────────────────


def load_env() -> None:
    """Load .env file if present (no dependency on python-dotenv)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_env(key: str) -> str:
    """Get a required environment variable, exit with message if missing."""
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"[ingest] ERROR: {key} not set. See .env.example", file=sys.stderr)
        sys.exit(1)
    return val


# ── embedding ──────────────────────────────────────────────────────────────


def load_embedding_model():
    """Load S-PubMedBert-MS-MARCO via sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    print(f"[ingest] Loading embedding model: {EMBEDDING_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBEDDING_MODEL)
    dt = time.time() - t0
    print(f"[ingest]   Model loaded in {dt:.1f}s")
    print(f"[ingest]   Embedding dim: {model.get_sentence_embedding_dimension()}")
    print(f"[ingest]   Max seq length: {model.max_seq_length}")

    dim = model.get_sentence_embedding_dimension()
    if dim != EMBEDDING_DIM:
        print(
            f"[ingest] WARNING: Expected {EMBEDDING_DIM}-dim, got {dim}-dim",
            file=sys.stderr,
        )

    return model


def embed_texts(model, texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Encode a list of texts into dense vectors.

    Returns list of float lists (not numpy) for JSON / pymilvus compat.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"[ingest] Embedding {len(texts)} texts on {device} (batch_size={batch_size})"
    )

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit-norm for IP metric
    )
    dt = time.time() - t0
    rate = len(texts) / dt if dt > 0 else 0
    print(f"[ingest]   Embedded in {dt:.1f}s ({rate:.0f} chunks/s)")

    # Convert to plain lists for pymilvus
    return [vec.tolist() for vec in embeddings]


# ── milvus ─────────────────────────────────────────────────────────────────


def connect_milvus():
    """Connect to Zilliz Cloud and return MilvusClient."""
    from pymilvus import MilvusClient

    endpoint = get_env("ZILLIZ_ENDPOINT")
    token = get_env("ZILLIZ_TOKEN")

    print(f"[ingest] Connecting to Zilliz Cloud...")
    print(f"[ingest]   Endpoint: {endpoint}")
    client = MilvusClient(uri=endpoint, token=token)
    print(f"[ingest]   Connected!")
    return client


def create_collection(client, recreate: bool = False) -> None:
    """Create the textbook_chunks collection with schema + BM25 function.

    Schema fields (from IMPLEMENTATION_PLAN):
        id             VARCHAR  PK
        text           VARCHAR  max=65535, enable_analyzer=True (BM25 input)
        dense_vector   FLOAT_VECTOR dim=768
        sparse_vector  SPARSE_FLOAT_VECTOR (BM25 auto-generated output)
        book           VARCHAR  (partition_key)
        chapter_num    INT16
        chapter_title  VARCHAR
        section        VARCHAR
        subsection     VARCHAR
        page_start     INT16
        page_end       INT16
        chunk_index    INT16
        content_type   VARCHAR

    Indexes:
        dense_vector   AUTOINDEX, metric=IP
        sparse_vector  AUTOINDEX, metric=BM25

    Function:
        BM25: text → sparse_vector
    """
    from pymilvus import DataType, Function, FunctionType

    exists = client.has_collection(COLLECTION_NAME)

    if exists and recreate:
        print(f"[ingest] Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)
        exists = False

    if exists:
        print(
            f"[ingest] Collection '{COLLECTION_NAME}' already exists, skipping creation"
        )
        return

    print(f"[ingest] Creating collection: {COLLECTION_NAME}")

    # ── schema ──
    schema = client.create_schema(auto_id=False)

    schema.add_field(
        field_name="id",
        datatype=DataType.VARCHAR,
        is_primary=True,
        max_length=128,
    )
    schema.add_field(
        field_name="text",
        datatype=DataType.VARCHAR,
        max_length=TEXT_MAX_LENGTH,
        enable_analyzer=True,
    )
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM,
    )
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR,
    )
    schema.add_field(
        field_name="book",
        datatype=DataType.VARCHAR,
        max_length=64,
    )
    schema.add_field(
        field_name="chapter_num",
        datatype=DataType.INT16,
    )
    schema.add_field(
        field_name="chapter_title",
        datatype=DataType.VARCHAR,
        max_length=256,
    )
    schema.add_field(
        field_name="section",
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name="subsection",
        datatype=DataType.VARCHAR,
        max_length=512,
    )
    schema.add_field(
        field_name="page_start",
        datatype=DataType.INT16,
    )
    schema.add_field(
        field_name="page_end",
        datatype=DataType.INT16,
    )
    schema.add_field(
        field_name="chunk_index",
        datatype=DataType.INT16,
    )
    schema.add_field(
        field_name="content_type",
        datatype=DataType.VARCHAR,
        max_length=64,
    )

    # ── BM25 function: text → sparse_vector ──
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    schema.add_function(bm25_function)

    # ── indexes ──
    index_params = client.prepare_index_params()

    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type="AUTOINDEX",
        metric_type="IP",
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_vector_index",
        index_type="AUTOINDEX",
        metric_type="BM25",
    )

    # ── create ──
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
    )

    print(f"[ingest]   Collection '{COLLECTION_NAME}' created with BM25 function")


def insert_chunks(
    client,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int,
) -> int:
    """Insert chunks into Milvus in batches.

    Returns total number of rows inserted.
    """
    total = len(chunks)
    inserted = 0

    print(f"[ingest] Inserting {total} chunks (batch_size={batch_size})")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_chunks = chunks[start:end]
        batch_embeds = embeddings[start:end]

        # Build row-based data for pymilvus
        data = []
        for chunk, embedding in zip(batch_chunks, batch_embeds):
            row = {
                "id": chunk["id"],
                "text": chunk["text"],
                "dense_vector": embedding,
                # sparse_vector is auto-generated by BM25 function — do NOT include
                "book": chunk["book"],
                "chapter_num": chunk["chapter_num"],
                "chapter_title": chunk["chapter_title"],
                "section": chunk.get("section", ""),
                "subsection": chunk.get("subsection", ""),
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "chunk_index": chunk["chunk_index"],
                "content_type": chunk["content_type"],
            }
            data.append(row)

        result = client.insert(
            collection_name=COLLECTION_NAME,
            data=data,
        )
        batch_inserted = result.get("insert_count", len(data))
        inserted += batch_inserted

        pct = inserted / total * 100
        print(f"[ingest]   {inserted}/{total} ({pct:.0f}%)", end="\r")

    print(f"[ingest]   Inserted {inserted}/{total} chunks" + " " * 20)
    return inserted


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed textbook chunks and insert into Zilliz Cloud (Milvus).",
    )
    parser.add_argument(
        "chunks_json",
        type=Path,
        help="Path to BATES.chunks.json from chunk_textbook.py",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Insertion batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=64,
        help="Embedding batch size for sentence-transformers (default: 64)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collection (WARNING: deletes existing data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed only, skip Milvus connection/insertion (for testing)",
    )
    args = parser.parse_args()

    # Load .env
    load_env()

    # Load chunks
    if not args.chunks_json.exists():
        print(f"[ingest] ERROR: {args.chunks_json} not found", file=sys.stderr)
        sys.exit(1)

    print(f"[ingest] Loading chunks from: {args.chunks_json}")
    chunks: list[dict[str, Any]] = json.loads(args.chunks_json.read_text())
    print(f"[ingest]   {len(chunks)} chunks loaded")

    # Validate chunk schema
    required = {
        "id",
        "text",
        "book",
        "chapter_num",
        "chapter_title",
        "page_start",
        "page_end",
        "chunk_index",
        "content_type",
    }
    sample = chunks[0]
    missing = required - set(sample.keys())
    if missing:
        print(
            f"[ingest] ERROR: Chunks missing required fields: {missing}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Embed
    model = load_embedding_model()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts, batch_size=args.embed_batch_size)

    assert len(embeddings) == len(chunks), "Embedding count mismatch"
    assert len(embeddings[0]) == EMBEDDING_DIM, (
        f"Expected {EMBEDDING_DIM}-dim, got {len(embeddings[0])}"
    )

    if args.dry_run:
        print(f"[ingest] Dry run complete — {len(embeddings)} embeddings generated")
        print(f"[ingest]   Sample embedding (first 5 dims): {embeddings[0][:5]}")
        return

    # Connect and insert
    client = connect_milvus()
    create_collection(client, recreate=args.recreate)

    t0 = time.time()
    n = insert_chunks(client, chunks, embeddings, batch_size=args.batch_size)
    dt = time.time() - t0
    print(f"[ingest] Insertion completed in {dt:.1f}s")

    # Verify
    stats = client.get_collection_stats(COLLECTION_NAME)
    row_count = stats.get("row_count", "unknown")
    print(f"[ingest] Collection '{COLLECTION_NAME}' now has {row_count} rows")

    print(f"\n[ingest] Done! {n} chunks ingested into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
