import os
import pickle
import warnings
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ updated import

# Suppress deprecated warnings for cleaner logs
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Flask app
app = Flask(__name__)

# Constants
PDF_PATH = "Nachiket_Shinde_Resume_v6.pdf"
INDEX_PATH = "portfolio_vector_index.pkl"

# Set API key for Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"

# Load LLM (lightweight model)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # fast & low-cost
    temperature=0.7
)

# Load or create vector index
def initialize_vector_index():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"❌ PDF not found: {PDF_PATH}")

    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            return pickle.load(f)

    # Load and process PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Limit number of chunks to save RAM
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)[:30]  # ✅ Limit to 30 chunks max

    # Use lightweight embeddings
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # ✅ use CPU to reduce GPU/mem usage
    )
    vector_index = FAISS.from_documents(docs, embedding)

    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(vector_index, f)

    return vector_index

# Initialize index + chain
vector_index = initialize_vector_index()
qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vector_index.as_retriever()
)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        result = qa_chain.invoke({"question": question})  # ✅ updated method
        return jsonify({
            "answer": result.get("answer", "No answer found."),
            "sources": result.get("sources", "")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "up"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # ✅ production-ready config
