import os
import pickle
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader

# Initialize Flask app
app = Flask(__name__)

# Constants
PDF_PATH = "Nachiket_shinde_resume_v6.pdf"
INDEX_PATH = "portfolio_vector_index.pkl"

# Load or create vector index
def initialize_vector_index():
    if os.path.exists(INDEX_PATH):
        with open(INDEX_PATH, 'rb') as f:
            return pickle.load(f)

    # Load and process PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Create embeddings and vector index
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_index = FAISS.from_documents(docs, embedding)

    # Save for future use
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump(vector_index, f)

    return vector_index

# Set API key for Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"

# Load LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# Initialize vector index and chain
vector_index = initialize_vector_index()
qa_chain = RetrievalQAWithSourcesChain.from_llm(
    llm=llm,
    retriever=vector_index.as_retriever()
)

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "Question not provided"}), 400

    try:
        result = qa_chain({"question": question}, return_only_outputs=True)
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
    app.run(debug=True, port=5000)
