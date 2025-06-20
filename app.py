import os
import logging
from threading import Lock
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

app = Flask(__name__)

# ========== Configuration ==========
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / 'static'

# Case-sensitive filename matching - MUST MATCH EXACTLY
PDF_FILENAME = "Nachiket_Shinde_Resume_v6.pdf"  # Note uppercase 'S' in Shinde
PDF_PATH = STATIC_DIR / PDF_FILENAME

# API configuration
QDRANT_URL = "https://b6bd2243-a196-48b4-abf7-9ee70021e4f4.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g3pa2VAIK87ueHxmKUhRf9uW1qVt_Z0I6JUpl7GqE1s"
GOOGLE_API_KEY = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

COLLECTION_NAME = "mychatbot"

# ========== Initialization ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_lock = Lock()
initialized = False
vectorstore = None
qa_chain = None

def find_pdf_file():
    """Find PDF file with case-insensitive check"""
    if PDF_PATH.exists():
        return PDF_PATH
    
    # Case-insensitive search
    for file in STATIC_DIR.iterdir():
        if file.name.lower() == PDF_FILENAME.lower():
            logger.warning(f"Found case-mismatched PDF: {file.name} (expected {PDF_FILENAME})")
            return file
    
    raise FileNotFoundError(
        f"PDF file not found. Looking for: {PDF_FILENAME}\n"
        f"Static directory contents: {[f.name for f in STATIC_DIR.iterdir()]}"
    )

def initialize_ai_components():
    global initialized, vectorstore, qa_chain
    
    with init_lock:
        if initialized:
            return
            
        logger.info("Starting AI components initialization...")
        
        try:
            # 1. Find PDF file with case-insensitive check
            pdf_file = find_pdf_file()
            logger.info(f"Using PDF file at: {pdf_file}")

            # 2. Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True
            )
            logger.info("Connected to Qdrant")

            # 3. Setup embeddings
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 4. Check if collection exists
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                if collection_info.points_count > 0:
                    logger.info(f"Using existing collection with {collection_info.points_count} vectors")
                else:
                    raise Exception("Empty collection - will recreate")
            except Exception:
                logger.info("Processing PDF and creating new collection...")
                
                # Load and process PDF
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                logger.info(f"Processed {len(split_docs)} document chunks")

                # Create collection
                qdrant_client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )

                # Store in Qdrant
                vectorstore = Qdrant(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embedding,
                )
                vectorstore.add_documents(split_docs)
                logger.info(f"Added documents to Qdrant")

            # Initialize QA chain
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embeddings=embedding,
            )
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7
            )
            
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )
            
            initialized = True
            logger.info("AI components initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

# ========== Routes ==========
@app.before_request
def handle_initialization():
    if not initialized:
        initialize_ai_components()

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "service": "Resume Chatbot API",
        "endpoints": {
            "/chat": "POST - Send chat messages",
            "/health": "GET - Service health check",
            "/debug": "GET - Debug information"
        }
    })

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "Message is required"}), 400
            
        result = qa_chain({"query": user_input})
        
        return jsonify({
            "answer": result["result"],
            "sources": [{
                "page": doc.metadata.get("page", "N/A"),
                "text": doc.page_content[:100] + "..."
            } for doc in result["source_documents"]]
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": "Processing failed", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    qdrant_status = False
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_status = client.get_collection(COLLECTION_NAME) is not None
    except Exception as e:
        logger.warning(f"Qdrant check failed: {str(e)}")
    
    return jsonify({
        "status": "healthy" if initialized else "initializing",
        "services": {
            "qdrant": qdrant_status,
            "pdf_loaded": STATIC_DIR.exists() and any(f.suffix == '.pdf' for f in STATIC_DIR.iterdir()),
            "ai_initialized": initialized
        }
    })

@app.route("/debug", methods=["GET"])
def debug_info():
    static_files = []
    if STATIC_DIR.exists():
        static_files = [{
            "name": f.name,
            "size": f.stat().st_size,
            "is_pdf": f.suffix.lower() == '.pdf'
        } for f in STATIC_DIR.iterdir()]
    
    return jsonify({
        "file_system": {
            "base_dir": str(BASE_DIR),
            "static_dir": str(STATIC_DIR),
            "static_dir_exists": STATIC_DIR.exists(),
            "static_files": static_files,
            "looking_for_pdf": PDF_FILENAME
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)