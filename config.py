"""
Centralized configuration for the Movie Recommender System.
All API keys and model settings are managed here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Embedding Model Settings ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- ChromaDB Settings ---
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION = "movies"

# --- LLM Settings ---
GEMINI_MODEL = "gemini-2.0-flash"

# --- App Settings ---
TOP_N_CANDIDATES = 15   # Number of candidates to pull from ChromaDB
TOP_N_DISPLAY = 5       # Number of final recommendations to show
