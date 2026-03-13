# app.py — Groq adapter version
from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
# removed langchain_openai import
# removed langchain chains imports (we'll call Groq directly)
from dotenv import load_dotenv
from src.prompt import *   # provides system_prompt
import os

# Import our Groq adapter you already created
from groq_adapter import chat_completion

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # use GROQ key now

# Keep Pinecone env var for the Pinecone client used by langchain_pinecone
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
# Also set GROQ_API_KEY in process env for groq_adapter (it reads os.environ inside)
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# load embeddings (your helper uses HuggingFace embeddings)
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
# Connect to existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create a retriever from the vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Helper: produce LLM answer using Groq and retrieved docs
def generate_answer_with_context(question: str, model: str = "llama-3.1-8b-instant", max_tokens: int = 512):
    # Retrieve relevant documents
    try:
        docs = retriever.get_relevant_documents(question)
    except AttributeError:
        # fallback: some vectorstore implementations use .get_relevant_documents or .get_relevant_documents
        # If as_retriever returns an object with .get_relevant_documents, use that; else try .retrieve
        if hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(question)
        else:
            docs = []

    # Combine the retrieved documents into a single context string
    context_parts = []
    for i, d in enumerate(docs):
        # d.page_content or d.content depending on Document shape
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        context_parts.append(f"Document {i+1}:\n{text}\n")

    context = "\n---\n".join(context_parts) if context_parts else "No retrieved context available."

    # Build messages for the chat model
    # system_prompt comes from src.prompt (existing)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer concisely with helpful medical guidance (not medical advice)."}
    ]

    # Call groq_adapter; it will use os.environ['GROQ_API_KEY']
    answer = chat_completion(messages, model=model, max_tokens=max_tokens)
    return answer

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg") or request.args.get("msg") or ""
    if not msg:
        return "No message provided", 400

    print("User question:", msg)
    try:
        response_text = generate_answer_with_context(msg)
    except Exception as e:
        print("Error generating answer:", e)
        return jsonify({"error": str(e)}), 500

    print("Response:", response_text)
    # Return plain text as before
    return str(response_text)

if __name__ == '__main__':
    # Use 0.0.0.0 and port 8080 to match previous behavior
    app.run(host="0.0.0.0", port=8080, debug=True)
