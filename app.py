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
PDF_FILENAME = "Nachiket_Shinde_Resume_v6.pdf"
PDF_PATH = STATIC_DIR / PDF_FILENAME

# API configuration
QDRANT_URL = "https://b6bd2243-a196-48b4-abf7-9ee70021e4f4.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g3pa2VAIK87ueHxmKUhRf9uW1qVt_Z0I6JUpl7GqE1s"
GOOGLE_API_KEY = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

COLLECTION_NAME = "mychatbot"
PORT = int(os.environ.get("PORT", 10000))  # Render uses port 10000

# ========== Initialization ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

init_lock = Lock()
initialized = False
vectorstore = None
qa_chain = None

def initialize_ai_components():
    global initialized, vectorstore, qa_chain
    
    with init_lock:
        if initialized:
            return
            
        logger.info("Starting AI components initialization...")
        
        try:
            # 1. Verify PDF exists
            if not PDF_PATH.exists():
                raise FileNotFoundError(
                    f"PDF not found at {PDF_PATH}\n"
                    f"Static directory contents: {[f.name for f in STATIC_DIR.iterdir()]}"
                )
            logger.info(f"Using PDF file at: {PDF_PATH}")

            # 2. Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                timeout=30  # Increased timeout for Render
            )
            logger.info("Connected to Qdrant")

            # 3. Setup embeddings
            embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

            # 4. Process documents only if collection is empty
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                if collection_info.points_count > 0:
                    logger.info(f"Using existing collection with {collection_info.points_count} vectors")
                else:
                    raise Exception("Collection empty - processing documents")
            except Exception:
                logger.info("Processing PDF and creating embeddings...")
                
                loader = PyPDFLoader(str(PDF_PATH))
                docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                
                qdrant_client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
                )
                
                vectorstore = Qdrant(
                    client=qdrant_client,
                    collection_name=COLLECTION_NAME,
                    embeddings=embedding,
                )
                vectorstore.add_documents(split_docs)
                logger.info(f"Added {len(split_docs)} documents to Qdrant")

            # Initialize QA chain
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7,
                timeout=60  # Increased timeout for Render
            )
            
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embeddings=embedding,
            )
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=vectorstore.as_retriever(),
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
            "/chat": {"method": "POST", "description": "Send chat messages"},
            "/health": {"method": "GET", "description": "Service health check"},
            "/debug": {"method": "GET", "description": "Debug information"}
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
    status = {
        "status": "healthy" if initialized else "initializing",
        "services": {
            "qdrant": False,
            "pdf_loaded": PDF_PATH.exists(),
            "ai_initialized": initialized
        }
    }
    
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=10
        )
        status["services"]["qdrant"] = client.get_collection(COLLECTION_NAME) is not None
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {str(e)}")
    
    return jsonify(status)

@app.route("/debug", methods=["GET"])
def debug_info():
    return jsonify({
        "file_system": {
            "base_dir": str(BASE_DIR),
            "static_dir": str(STATIC_DIR),
            "static_dir_exists": STATIC_DIR.exists(),
            "static_files": [f.name for f in STATIC_DIR.iterdir()],
            "looking_for_pdf": PDF_FILENAME,
            "pdf_exists": PDF_PATH.exists()
        },
        "environment": {
            "port": PORT,
            "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
            "render": os.environ.get("RENDER", "false")
        }
    })

if __name__ == "__main__":
    # For Render deployment
    from waitress import serve
    logger.info(f"Starting server on port {PORT}")
    initialize_ai_components()  # Initialize before serving
    serve(app, host="0.0.0.0", port=PORT)