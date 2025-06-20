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
import logging
from threading import Lock

app = Flask(__name__)

# Configuration with API keys (FOR TESTING ONLY - REPLACE WITH YOUR OWN KEYS)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, 'static', 'Nachiket_shinde_Resume_v6.pdf')
COLLECTION_NAME = "mychatbot"

# Qdrant Cloud Configuration
QDRANT_URL = "https://b6bd2243-a196-48b4-abf7-9ee70021e4f4.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g3pa2VAIK87ueHxmKUhRf9uW1qVt_Z0I6JUpl7GqE1s"

# Google Gemini Configuration
GOOGLE_API_KEY = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe initialization
init_lock = Lock()
initialized = False
vectorstore = None
qa_chain = None

def initialize_ai_components():
    global initialized, vectorstore, qa_chain
    
    with init_lock:
        if initialized:
            return
            
        logger.info("Initializing AI components for the first time...")
        
        try:
            # 1. Verify PDF exists
            if not os.path.exists(PDF_PATH):
                raise FileNotFoundError(f"PDF not found at {PDF_PATH}")
            logger.info(f"Found PDF at {PDF_PATH}")

            # 2. Initialize Qdrant client
            qdrant_client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True
            )
            logger.info("Connected to Qdrant")

            # 3. Check if collection exists and has data
            try:
                collection_info = qdrant_client.get_collection(COLLECTION_NAME)
                if collection_info.points_count > 0:
                    logger.info(f"Collection {COLLECTION_NAME} exists with {collection_info.points_count} vectors")
                else:
                    raise Exception("Collection exists but is empty")
            except Exception as e:
                logger.info(f"Processing PDF: {str(e)}")
                
                # Load and split PDF
                loader = PyPDFLoader(PDF_PATH)
                docs = loader.load()
                splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                split_docs = splitter.split_documents(docs)
                logger.info(f"Split into {len(split_docs)} chunks")

                # Create embeddings
                embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

                # Create/recreate collection
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
                logger.info(f"Added {len(split_docs)} documents to Qdrant")

            # Initialize QA chain
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.7
            )
            
            vectorstore = Qdrant(
                client=qdrant_client,
                collection_name=COLLECTION_NAME,
                embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            )
            
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )
            
            initialized = True
            logger.info("Initialization complete")

        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            raise

# Modern Flask initialization approach
@app.before_request
def before_first_request():
    if not initialized:
        initialize_ai_components()

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat queries"""
    try:
        if not initialized:
            initialize_ai_components()
            
        user_input = request.json.get("message", "")
        if not user_input:
            return jsonify({"error": "No message provided"}), 400
            
        result = qa_chain({"query": user_input})
        
        return jsonify({
            "answer": result["result"],
            "sources": [{
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "N/A")
            } for doc in result["source_documents"]]
        })
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        qdrant_ready = client.get_collection(COLLECTION_NAME) is not None
    except:
        qdrant_ready = False
        
    return jsonify({
        "status": "healthy",
        "qdrant_ready": qdrant_ready,
        "initialized": initialized,
        "pdf_exists": os.path.exists(PDF_PATH)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)