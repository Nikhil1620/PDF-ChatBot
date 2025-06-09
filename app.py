from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import tempfile
import shutil
from datetime import datetime
import uuid
import chromadb
from chromadb.config import Settings

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Ensure required environment variables are set




# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ChromaDB configuration
CHROMA_DB_DIR = "chroma_db"
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

# Global variables to store sessions
active_sessions = {}

def allowed_file(filename):
    """Check if uploaded file is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_pdf_text(pdf_files):
    """Extract text from PDF files"""
    text = ""
    try:
        for pdf_file in pdf_files:
            pdf_reader = PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # Reduced chunk size for better ChromaDB performance
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_chroma_vectorstore(text_chunks, session_id):
    """Create ChromaDB vector store"""
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create collection name (ChromaDB collection names must be alphanumeric)
        collection_name = f"session_{session_id.replace('-', '_')}"
        
        # Create ChromaDB client with persistent storage
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Delete collection if it exists (for fresh start)
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        # Create Chroma vector store
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=CHROMA_DB_DIR
        )
        
        # Add documents to the vector store
        vectorstore.add_texts(
            texts=text_chunks,
            metadatas=[{"chunk_id": i, "session_id": session_id} for i in range(len(text_chunks))]
        )
        
        # Store session info
        active_sessions[session_id] = {
            'collection_name': collection_name,
            'created_at': datetime.now(),
            'chunks_count': len(text_chunks),
            'client': client,
            'vectorstore': vectorstore
        }
        
        return True
        
    except Exception as e:
        raise Exception(f"Error creating ChromaDB vector store: {str(e)}")

def get_chroma_vectorstore(session_id):
    """Get existing ChromaDB vector store"""
    try:
        if session_id not in active_sessions:
            return None
            
        session_info = active_sessions[session_id]
        
        # If vectorstore is already loaded, return it
        if 'vectorstore' in session_info and session_info['vectorstore']:
            return session_info['vectorstore']
        
        # Otherwise, recreate the connection
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        collection_name = session_info['collection_name']
        
        client = chromadb.PersistentClient(
            path=CHROMA_DB_DIR,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            client=client,
            persist_directory=CHROMA_DB_DIR
        )
        
        # Update session info
        active_sessions[session_id]['client'] = client
        active_sessions[session_id]['vectorstore'] = vectorstore
        
        return vectorstore
        
    except Exception as e:
        raise Exception(f"Error loading ChromaDB vector store: {str(e)}")

def get_conversational_chain():
    """Create conversational chain"""
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'PDF Chatbot API with ChromaDB is running',
        'timestamp': datetime.now().isoformat(),
        'database': 'ChromaDB',
        'active_sessions': len(active_sessions)
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process PDF files"""
    try:
        # Check if files are present in request
        if 'files' not in request.files:
            return jsonify({
                'error': 'No files provided',
                'message': 'Please upload at least one PDF file'
            }), 400
        
        files = request.files.getlist('files')
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({
                'error': 'No files selected',
                'message': 'Please select at least one PDF file'
            }), 400
        
        # Validate file types
        pdf_files = []
        for file in files:
            if file and allowed_file(file.filename):
                pdf_files.append(file)
            else:
                return jsonify({
                    'error': 'Invalid file type',
                    'message': f'File {file.filename} is not a PDF file'
                }), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Process PDF files
        temp_files = []
        try:
            # Save uploaded files temporarily
            for file in pdf_files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                file.save(temp_file.name)
                temp_files.append(temp_file.name)
            
            # Extract text from PDFs
            raw_text = get_pdf_text(temp_files)
            
            if not raw_text.strip():
                return jsonify({
                    'error': 'No text found',
                    'message': 'No readable text found in the uploaded PDF files'
                }), 400
            
            # Split text into chunks
            text_chunks = get_text_chunks(raw_text)
            
            # Create ChromaDB vector store
            create_chroma_vectorstore(text_chunks, session_id)
            
            return jsonify({
                'success': True,
                'message': 'Files processed successfully with ChromaDB',
                'session_id': session_id,
                'files_processed': len(pdf_files),
                'text_chunks': len(text_chunks),
                'database': 'ChromaDB'
            })
            
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
    except Exception as e:
        return jsonify({
            'error': 'Processing failed',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with the uploaded documents using ChromaDB"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide JSON data with question and session_id'
            }), 400
        
        question = data.get('question', '').strip()
        session_id = data.get('session_id', '').strip()
        
        if not question:
            return jsonify({
                'error': 'No question provided',
                'message': 'Please provide a question'
            }), 400
        
        if not session_id:
            return jsonify({
                'error': 'No session ID provided',
                'message': 'Please provide a valid session_id'
            }), 400
        
        # Check if session exists
        if session_id not in active_sessions:
            return jsonify({
                'error': 'Invalid session',
                'message': 'Session not found. Please upload PDF files first.'
            }), 404
        
        # Get ChromaDB vector store
        vectorstore = get_chroma_vectorstore(session_id)
        
        if not vectorstore:
            return jsonify({
                'error': 'Vector store not found',
                'message': 'Could not load the document store for this session'
            }), 404
        
        # Search for relevant documents
        docs = vectorstore.similarity_search(question, k=4)
        
        if not docs:
            return jsonify({
                'answer': 'No relevant information found in the uploaded documents.',
                'question': question,
                'session_id': session_id,
                'database': 'ChromaDB'
            })
        
        # Get conversational chain and generate response
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": question},
            return_only_outputs=True
        )
        
        return jsonify({
            'answer': response["output_text"],
            'question': question,
            'session_id': session_id,
            'sources_found': len(docs),
            'database': 'ChromaDB'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Chat processing failed',
            'message': str(e)
        }), 500

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all active sessions"""
    try:
        sessions_info = []
        for session_id, info in active_sessions.items():
            sessions_info.append({
                'session_id': session_id,
                'created_at': info['created_at'].isoformat(),
                'collection_name': info['collection_name'],
                'chunks_count': info['chunks_count']
            })
        
        return jsonify({
            'sessions': sessions_info,
            'total_sessions': len(sessions_info),
            'database': 'ChromaDB'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to get sessions',
            'message': str(e)
        }), 500

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    """Delete a specific session and its ChromaDB collection"""
    try:
        if session_id not in active_sessions:
            return jsonify({
                'error': 'Session not found',
                'message': f'Session {session_id} does not exist'
            }), 404
        
        session_info = active_sessions[session_id]
        collection_name = session_info['collection_name']
        
        # Delete ChromaDB collection
        try:
            client = chromadb.PersistentClient(
                path=CHROMA_DB_DIR,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            client.delete_collection(collection_name)
        except Exception as e:
            print(f"Warning: Could not delete ChromaDB collection {collection_name}: {e}")
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        return jsonify({
            'success': True,
            'message': f'Session {session_id} and its ChromaDB collection deleted successfully'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Failed to delete session',
            'message': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_documents():
    """Advanced search in documents with similarity scores"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide JSON data with query and session_id'
            }), 400
        
        query = data.get('query', '').strip()
        session_id = data.get('session_id', '').strip()
        k = data.get('k', 5)  # Number of results to return
        
        if not query or not session_id:
            return jsonify({
                'error': 'Missing parameters',
                'message': 'Please provide both query and session_id'
            }), 400
        
        if session_id not in active_sessions:
            return jsonify({
                'error': 'Invalid session',
                'message': 'Session not found'
            }), 404
        
        # Get vector store
        vectorstore = get_chroma_vectorstore(session_id)
        
        if not vectorstore:
            return jsonify({
                'error': 'Vector store not found',
                'message': 'Could not load the document store'
            }), 404
        
        # Perform similarity search with scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for doc, score in docs_with_scores:
            results.append({
                'content': doc.page_content,
                'similarity_score': float(score),
                'metadata': doc.metadata
            })
        
        return jsonify({
            'results': results,
            'query': query,
            'session_id': session_id,
            'total_results': len(results),
            'database': 'ChromaDB'
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Search failed',
            'message': str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large', 
        'message': 'File size exceeds 16MB limit'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on the server'
    }), 500

if __name__ == '__main__':
    print("🚀 Starting PDF Chatbot API with ChromaDB...")
    print("📚 Upload PDFs to /api/upload")
    print("💬 Chat at /api/chat")
    print("🔍 Search documents at /api/search")
    print("🏥 Check health at /api/health")
    print("📁 ChromaDB storage location:", CHROMA_DB_DIR)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)