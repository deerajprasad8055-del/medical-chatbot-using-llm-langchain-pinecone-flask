# store_index_upsert.py
import os
from uuid import uuid4
from math import ceil
from dotenv import load_dotenv

# your helper provides: load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings

# Pinecone (serverless SDK)
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY not set in environment. Run: set PINECONE_API_KEY=...")

# create client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# create index if not exists (safe)
if not pc.has_index(index_name):
    print(f"Index '{index_name}' not found. Creating...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
else:
    print(f"Index '{index_name}' already exists.")

index = pc.Index(index_name)

# 1) Load and prepare chunks
print("Loading PDFs...")
extracted_data = load_pdf_file(data="data/")
print(f"Loaded {len(extracted_data)} documents.")

print("Filtering metadata...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting into chunks...")
text_chunks = text_split(filter_data)
print(f"Created {len(text_chunks)} text chunks.")

if len(text_chunks) == 0:
    print("No text chunks found. Exiting.")
    raise SystemExit(0)

# 2) Get embeddings wrapper
embeddings = download_hugging_face_embeddings()

# 3) Build upsert batches and upsert
batch_size = 50   # tune this (50-200) based on memory/network
total = len(text_chunks)
batches = ceil(total / batch_size)
print(f"Will upload {total} vectors in {batches} batches (batch_size={batch_size}).")

for i in range(0, total, batch_size):
    slice_chunks = text_chunks[i : i + batch_size]

    # embed_documents expects a list; returns list of vectors
    vectors = embeddings.embed_documents(slice_chunks)  # each element is a list[float]

    # prepare upsert payload: list of (id, values, metadata) or dicts depending on SDK
    # The Serverless SDK index.upsert accepts a list of dicts: {"id": ..., "values": ..., "metadata": {...}}
    upsert_items = []
    for chunk, vec in zip(slice_chunks, vectors):
        # Use a deterministic or random id. We'll use uuid4 here.
        vid = str(uuid4())
        meta = {"source": getattr(chunk, "metadata", {}).get("source", ""), "text": getattr(chunk, "page_content", "")[:200]}
        upsert_items.append({"id": vid, "values": vec, "metadata": meta})

    print(f"Uploading batch {i//batch_size + 1}/{batches} ({len(upsert_items)} vectors)...")
    index.upsert(vectors=upsert_items)

print("All batches uploaded. Waiting a few seconds for index to update...")

# 4) Print index stats
stats = index.describe_index_stats()
print("Index stats:", stats)
print("Done.")
