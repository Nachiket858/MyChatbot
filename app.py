import os
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

# Initialize Flask app
app = Flask(__name__)

# Constants
PDF_PATH = "Nachiket_shinde_resume_v6.pdf"
CHROMA_PATH = "chroma_db"

# Set API key for Gemini
os.environ["GOOGLE_API_KEY"] = "AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8"

# Load LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7
)

# Load or create vector index using ChromaDB
def initialize_vector_index():
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(CHROMA_PATH):
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

    # Load and process PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(documents)

    # Create Chroma vector store
    vector_index = Chroma.from_documents(
        docs,
        embedding,
        persist_directory=CHROMA_PATH
    )
    vector_index.persist()
    return vector_index

# Initialize vector store and retrieval chain
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
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
