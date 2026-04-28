"""
build_vectordb.py — Upgraded Data Pipeline
==========================================
Replaces the old main.py pipeline:
  - Same merge & cleaning logic for Hollywood + Indian datasets
  - Uses all-MiniLM-L6-v2 for semantic embeddings (instead of CountVectorizer)
  - Stores vectors in ChromaDB (instead of a 2.2 GB pickle file)

Run this ONCE (or whenever your CSV data changes):
    python build_vectordb.py
"""

import sys
import io

# Fix Windows console encoding for Unicode output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import ast
import os
import shutil
from sentence_transformers import SentenceTransformer
import chromadb
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, CHROMA_COLLECTION

# ============================================================
# STEP 1: LOAD & MERGE HOLLYWOOD DATA (unchanged from main.py)
# ============================================================
print("[1/7] Loading Hollywood datasets...")
movies = pd.read_csv('data/movies.csv')
links = pd.read_csv('data/links.csv')
tmdb_movies = pd.read_csv('data/tmdb_5000_movies.csv')
tmdb_credits = pd.read_csv('data/tmdb_5000_credits.csv')

# Merge TMDB files
tmdb = tmdb_movies.merge(tmdb_credits, left_on='id', right_on='movie_id')

# Link MovieLens to TMDB
ml_data = movies.merge(links, on='movieId')
ml_data.dropna(subset=['tmdbId'], inplace=True)
ml_data['tmdbId'] = ml_data['tmdbId'].astype(int)

# Hollywood Master Dataset
hw_df = ml_data.merge(tmdb, left_on='tmdbId', right_on='id')

# ============================================================
# STEP 2: CLEANING FUNCTIONS (unchanged from main.py)
# ============================================================
def convert_to_list(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            L.append(i['name'])
    except:
        return []
    return L

def get_top_3(obj):
    L = []
    counter = 0
    try:
        for i in ast.literal_eval(obj):
            if counter < 3:
                L.append(i['name'])
                counter += 1
            else: break
    except:
        return []
    return L

def fetch_director(obj):
    L = []
    try:
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
    except:
        return []
    return L

# Defensive collapse function to handle NaNs/Floats
def collapse(L):
    L1 = []
    for i in L:
        if isinstance(i, str) and i.lower() != 'nan':
            L1.append(i.replace(" ",""))
    return L1

# ============================================================
# STEP 3: APPLY CLEANING TO HOLLYWOOD (unchanged from main.py)
# ============================================================
print("[2/7] Cleaning Hollywood data...")
hw_df['genres'] = hw_df['genres_y'].apply(convert_to_list)
hw_df['keywords'] = hw_df['keywords'].apply(convert_to_list)
hw_df['cast'] = hw_df['cast'].apply(get_top_3)
hw_df['crew'] = hw_df['crew'].apply(fetch_director)

# Remove spaces from names
for col in ['genres', 'keywords', 'cast', 'crew']:
    hw_df[col] = hw_df[col].apply(collapse)

hw_df['overview'] = hw_df['overview'].fillna('').apply(lambda x: x.lower().split())

# Create Hollywood Tags
hw_df['tags'] = hw_df['overview'] + hw_df['genres'] + hw_df['keywords'] + hw_df['cast'] + hw_df['crew']
hw_final = hw_df[['tmdbId', 'title_y', 'tags']].rename(columns={'title_y': 'title'})

# ============================================================
# STEP 4: LOAD & CLEAN INDIAN DATA (unchanged from main.py)
# ============================================================
print("[3/7] Loading Indian datasets...")
indian_meta = pd.read_csv('data/movies_data.csv')
indian_desc = pd.read_csv('data/IMDB_10000.csv')

# Use OUTER join to keep ALL movies from both files
indian_full = indian_meta.merge(indian_desc, left_on='Name', right_on='title', how='outer')

# Fill NaNs so the logic doesn't crash
indian_full['title'] = indian_full['title'].fillna(indian_full['Name'])
indian_full['desc'] = indian_full['desc'].fillna('')
indian_full['Genre'] = indian_full['Genre'].fillna('')
for col in ['Actor 1', 'Actor 2', 'Actor 3', 'Director']:
    indian_full[col] = indian_full[col].fillna('')

# Process Indian Genres, Cast, and Director
indian_full['genres'] = indian_full['Genre'].apply(lambda x: [i.strip().replace(" ","") for i in str(x).split(',')] if x != '' else [])
indian_full['cast'] = indian_full.apply(lambda row: [row['Actor 1'], row['Actor 2'], row['Actor 3']], axis=1).apply(collapse)
indian_full['crew'] = indian_full['Director'].apply(lambda x: [str(x).replace(" ","")] if x != '' else [])

# Process Indian Description
indian_full['desc_list'] = indian_full['desc'].apply(lambda x: x.lower().split())

# Create Indian Tags
indian_full['tags'] = indian_full['desc_list'] + indian_full['genres'] + indian_full['cast'] + indian_full['crew']
indian_final = indian_full[['title', 'tags']].copy()
indian_final['tmdbId'] = 0

# ============================================================
# STEP 5: COMBINE EVERYTHING (unchanged from main.py)
# ============================================================
new_df = pd.concat([hw_final, indian_final], ignore_index=True)

# Convert tags from lists back to strings
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# Drop any rows with empty tags
new_df = new_df[new_df['tags'].str.strip() != ''].reset_index(drop=True)

# Drop duplicate titles (keep first occurrence)
new_df = new_df.drop_duplicates(subset='title', keep='first').reset_index(drop=True)

print(f"[4/7] Total Combined Database: {len(new_df)} unique movies ready!")

# ============================================================
# STEP 6: GENERATE SEMANTIC EMBEDDINGS (NEW — replaces CountVectorizer)
# ============================================================
print(f"[5/7] Loading embedding model: {EMBEDDING_MODEL}...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("[6/7] Generating 384-dimensional embeddings for all movies...")
print("      (This may take a few minutes on first run...)")
embeddings = model.encode(
    new_df['tags'].tolist(),
    show_progress_bar=True,
    batch_size=256
)
print(f"      Generated embeddings: shape {embeddings.shape}")

# ============================================================
# STEP 7: STORE IN CHROMADB (NEW — replaces pickle)
# ============================================================
print(f"[7/7] Building ChromaDB at: {CHROMA_DB_PATH}")

# Clear old database if it exists (for clean rebuild)
if os.path.exists(CHROMA_DB_PATH):
    shutil.rmtree(CHROMA_DB_PATH)
    print("   (Cleared previous database)")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION,
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# ChromaDB has a batch limit, so we insert in chunks
BATCH_SIZE = 500
total = len(new_df)

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_df = new_df.iloc[start:end]
    batch_embeddings = embeddings[start:end]

    collection.add(
        ids=[str(i) for i in range(start, end)],
        embeddings=batch_embeddings.tolist(),
        metadatas=[
            {
                "title": str(row['title']),
                "tmdbId": int(row['tmdbId']),
                "tags": str(row['tags'])[:500]  # Truncate long tags for metadata
            }
            for _, row in batch_df.iterrows()
        ],
        documents=batch_df['tags'].tolist()
    )
    print(f"   Inserted batch {start}-{end} of {total}")

print(f"\n=== ChromaDB built successfully! ===")
print(f"    Collection '{CHROMA_COLLECTION}' contains {collection.count()} movies")
print(f"    Storage path: {os.path.abspath(CHROMA_DB_PATH)}")
print(f"\n>>> You can now run: streamlit run app.py")
