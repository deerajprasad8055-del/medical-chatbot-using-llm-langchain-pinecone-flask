from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
docs = load_pdf_file("data/")
docs = filter_to_minimal_docs(docs)
chunks = text_split(docs)[:2]
emb = download_hugging_face_embeddings()
vecs = emb.embed_documents([chunks[0]])
print("Chunks found:", len(chunks), "Vector length:", len(vecs[0]))
