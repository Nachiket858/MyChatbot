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

# Constants
PDF_PATH = "static/Nachiket_shinde_Resume_v6.pdf"
COLLECTION_NAME = "mychatbot"


QDRANT_URL = "https://b6bd2243-a196-48b4-abf7-9ee70021e4f4.us-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.g3pa2VAIK87ueHxmKUhRf9uW1qVt_Z0I6JUpl7GqE1s"



# Gemini API key (set it as an environment variable or paste directly)
os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"

# Load PDF and split
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = splitter.split_documents(docs)

# Initialize embeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Qdrant client and collection
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# Ensure collection exists
qdrant.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Push PDF embeddings to Qdrant
vectorstore = Qdrant(
    client=qdrant,
    collection_name=COLLECTION_NAME,
    embeddings=embedding,
)
vectorstore.add_documents(split_docs)

# Setup Gemini model
llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyA3GbDc39XAxR-4fVHII3D0mf_5Ftf7ph8",
        temperature=0.7
    )

# QA Chain using Gemini + Qdrant
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No message provided."}), 400
    result = qa_chain({"query": user_input})
    return jsonify({
        "answer": result["result"],
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
