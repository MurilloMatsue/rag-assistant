import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
TOP_K           = 4

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.1-8b-instant"

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR        = os.path.join(BASE_DIR, "docs")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")