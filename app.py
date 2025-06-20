import os
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

app = Flask(__name__)

# Configuration
PDF_PATH = os.path.join(os.path.dirname(__file__), 'static', 'Nachiket_shinde_Resume_v6.pdf')
COLLECTION_NAME = "mychatbot"

# Hardcoded keys (used for testing only)
QDRANT_URL = "https://b6bd2243-a196-48b4-abf7-9ee70021e4f4.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g3pa2VAIK87ueHxmKUhRf9uW1qVt_Z0I6JUpl7GqE1s"
GOOGLE_API_KEY = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"

# Gemini key setup for LangChain
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize components
vectorstore = None
qa_chain = None

def initialize_ai_components():
    global vectorstore, qa_chain

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    # 1. Load and process PDF
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    # 2. Embedding model
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Qdrant setup
    qdrant_client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    try:
        qdrant_client.get_collection(COLLECTION_NAME)
    except Exception:
        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    # 4. Create vectorstore
    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding,
    )
    vectorstore.add_documents(split_docs)

    # 5. Load Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        google_api_key=GOOGLE_API_KEY
    )

    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )

# API Endpoints
@app.route("/chat", methods=["POST"])
def chat():
    try:
        if qa_chain is None:
            initialize_ai_components()

        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400

        result = qa_chain({"query": user_input})
        return jsonify({
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "qdrant_ready": QDRANT_URL is not None,
        "google_ready": GOOGLE_API_KEY is not None,
        "pdf_exists": os.path.exists(PDF_PATH)
    })

@app.route("/reload", methods=["POST"])
def reload_data():
    try:
        initialize_ai_components()
        return jsonify({"status": "data reloaded"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
