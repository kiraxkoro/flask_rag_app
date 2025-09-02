import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub

load_dotenv()  # loads values from .env file

# --- Flask App Initialization ---
app = Flask(__name__)
# Create a folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Global Variables for RAG Components ---
vector_store = None
rag_chain = None
llm = None
current_model_name = ""

# --- Model Configuration ---
def get_gemini_llm():
    """Initializes the Google Gemini Pro model."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    return GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)  # Updated to free-tier compatible model

def get_flan_t5_llm():
    """Initializes the Google Flan-T5 model from Hugging Face."""
    if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set.")
    return HuggingFaceHub(
        repo_id="google/flan-t5-large",
        task="text2text-generation",  # Add this to specify the task
        model_kwargs={"temperature": 0.5}
    )

MODELS = {
    "gemini": get_gemini_llm,
    "flan-t5": get_flan_t5_llm,
}

def load_model(model_name: str):
    """Loads a model by name and sets it globally."""
    global llm, current_model_name
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODELS.keys())}")
    
    print(f"Loading model: {model_name}...")
    llm = MODELS[model_name]()
    current_model_name = model_name
    print(f"Model {model_name} loaded successfully.")

# --- API Endpoints ---
@app.route('/')
def home():
    return """
    <h1>RAG API Server</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><b>POST /input_files</b> - Upload PDF/TXT files</li>
        <li><b>POST /query</b> - Ask questions about your documents</li>
        <li><b>POST /switch_model</b> - Switch between models (gemini/flan-t5)</li>
    </ul>
    <p>Use tools like curl, Postman, or Thunder Client to interact with the API.</p>
    """

@app.route('/input_files', methods=['POST'])
def input_files_handler():
    """API Endpoint to upload files, process them, and create a RAG chain."""
    global vector_store, rag_chain

    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400

    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No selected files"}), 400

    # 1. Load Documents
    docs = []
    for file in files:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file.filename.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            continue
        docs.extend(loader.load())

    if not docs:
        return jsonify({"error": "No processable documents found (only .pdf and .txt are supported)."}), 400

    # 2. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # 3. Create Vector Embeddings and Store in FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents=splits, embedding=embeddings)

    # 4. Create the RAG Chain
    retriever = vector_store.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    return jsonify({
        "message": f"Successfully processed {len(docs)} documents.",
        "total_chunks": len(splits),
        "vector_store": "FAISS in-memory",
        "current_model": current_model_name
    }), 200

@app.route('/query', methods=['POST'])
def query_handler():
    """API Endpoint to ask a question to the RAG chain."""
    if not rag_chain:
        return jsonify({"error": "RAG chain not initialized. Please upload files to '/input_files' first."}), 400

    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "Query not provided in request body."}), 400
    
    try:
        result = rag_chain.invoke({"query": query})
        return jsonify({"answer": result['result']})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/switch_model', methods=['POST'])
def switch_model_handler():
    """API Endpoint to switch the backend language model."""
    global rag_chain
    
    data = request.get_json()
    new_model_name = data.get('model_name')

    if not new_model_name:
        return jsonify({"error": "model_name not provided in request body."}), 400
    
    try:
        load_model(new_model_name)
        if vector_store:
            retriever = vector_store.as_retriever()
            rag_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            message = f"Model switched to {new_model_name}. Existing RAG chain has been updated."
        else:
            message = f"Model switched to {new_model_name}. Upload files to create a RAG chain."
            
        return jsonify({"message": message}), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        print("--- RAG Server Initializing ---")
        load_model("gemini") # Default to Gemini
    except Exception as e:
        print(f"Could not load default model on startup: {e}")
        print("Please set your API keys and use the /switch_model endpoint.")
    
    app.run(debug=True, port=5001)