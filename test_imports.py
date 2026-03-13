# test_imports.py — checks required imports
try:
    from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    print("All imports OK")
except Exception as e:
    print("Import error:", e)
