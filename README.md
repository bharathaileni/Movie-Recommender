# 🎬 Movie Recommender System — AI-Powered

An intelligent movie recommendation system that combines **content-based filtering** with **LLM-powered mood analysis** to deliver personalized movie suggestions across **14,800+ Hollywood & Indian films**.

## ✨ Key Features

| Feature | Description |
|---|---|
| **🎯 Classic Mode** | Select a movie you liked → get 5 semantically similar recommendations |
| **💬 Mood Mode** | Describe your mood in natural language → AI interprets it and recommends movies |
| **🌐 Cross-Industry** | Unified database covering Hollywood (TMDB 5000) + Indian cinema (Bollywood/Tollywood) |
| **🧠 Semantic Search** | 384-dimensional sentence embeddings via `all-MiniLM-L6-v2` for deep content understanding |
| **🤖 Gemini LLM Re-ranking** | Google Gemini re-ranks semantic search results with conversational explanations |
| **🖼️ Rich Poster UI** | Auto-fetched movie posters from TMDB API with hover effects and card layouts |

## 🏗️ Architecture

```
User Input (Movie Title / Mood Text)
        │
        ▼
┌─────────────────────────────────┐
│   Sentence Transformer Encoder  │  ← all-MiniLM-L6-v2 (384-dim)
│   (Semantic Embedding)          │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│        ChromaDB Vector Store    │  ← 14,800+ movies indexed
│   (Cosine Similarity Search)    │
└──────────────┬──────────────────┘
               │
        ┌──────┴──────┐
        │             │
   Classic Mode   Mood Mode
        │             │
        ▼             ▼
  Top 5 Similar   Google Gemini
    Movies        LLM Re-ranking
        │         + Conversational
        │           Response
        ▼             │
┌─────────────────────┴───────────┐
│        Streamlit Frontend       │
│   (Posters via TMDB API)        │
└─────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit (with custom CSS for gradient headers, movie cards, hover effects)
- **Vector Database**: ChromaDB (persistent storage, cosine similarity, replaces 2.2 GB pickle)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`) — 384-dimensional semantic vectors
- **LLM**: Google Gemini (`gemini-2.0-flash`) — mood interpretation & conversational re-ranking
- **Data Processing**: Pandas, AST parsing for nested JSON columns
- **Poster API**: TMDB (The Movie Database) REST API
- **ML (Legacy)**: Scikit-Learn CountVectorizer + Cosine Similarity (preserved in `main.py`)

## 📂 Project Structure

```
Movie-Recommender/
├── app.py               # Main Streamlit app (Dual-mode: Classic + Mood)
├── build_vectordb.py    # Data pipeline: CSV → cleaned tags → ChromaDB
├── llm_service.py       # Gemini LLM integration for mood-based re-ranking
├── config.py            # Centralized configuration (API keys, model settings)
├── main.py              # Original ML pipeline (CountVectorizer + cosine similarity)
├── app_legacy.py        # Original Streamlit UI (pickle-based)
├── requirements.txt     # Python dependencies
├── .gitignore           # Ignores venv, .env, data CSVs, chroma_db, pkl files
└── data/                # Raw datasets (gitignored)
    ├── tmdb_5000_movies.csv
    ├── tmdb_5000_credits.csv
    ├── movies.csv           # MovieLens
    ├── links.csv            # MovieLens → TMDB ID mapping
    ├── ratings.csv          # MovieLens ratings
    ├── movies_data.csv      # Indian cinema metadata
    └── IMDB_10000.csv       # Indian cinema descriptions
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [TMDB API Key](https://www.themoviedb.org/settings/api)
- [Google Gemini API Key](https://aistudio.google.com/apikey)

### Installation

```bash
# Clone the repository
git clone https://github.com/bharathaileni/Movie-Recommender.git
cd Movie-Recommender

# Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### Build the Vector Database

```bash
python build_vectordb.py
```

This processes all 7 CSV datasets, generates semantic embeddings, and stores them in ChromaDB (~15 min on first run).

### Run the App

```bash
streamlit run app.py
```

## 📊 Data Pipeline

The `build_vectordb.py` script performs a 7-step pipeline:

1. **Load** Hollywood datasets (TMDB 5000 + MovieLens)
2. **Clean** — Parse JSON columns (`genres`, `cast`, `crew`, `keywords`) using `ast.literal_eval`
3. **Merge** — Link MovieLens to TMDB via `tmdbId` mapping
4. **Load** Indian cinema datasets (10,000+ Bollywood/Tollywood films)
5. **Combine** — Unify both datasets with deduplication (14,800+ unique movies)
6. **Embed** — Generate 384-dim vectors using `all-MiniLM-L6-v2`
7. **Store** — Batch-insert into ChromaDB with cosine similarity indexing

## 🤖 How Mood Mode Works

1. User types a natural language mood description
2. The mood text is encoded into the same 384-dim embedding space
3. ChromaDB returns the top 15 semantically closest movies
4. Google Gemini receives these candidates and:
   - Re-ranks them based on mood fit
   - Picks the top 5 best matches
   - Generates a conversational response with match ratings (⭐ to ⭐⭐⭐⭐⭐)
5. If the LLM API is unavailable, the system gracefully falls back to raw ChromaDB results

## 📈 Evolution of the Project

| Version | Approach | Storage | Drawback |
|---|---|---|---|
| **v1 (Legacy)** | CountVectorizer + Cosine Similarity | 2.2 GB pickle file | Bag-of-words, no semantic understanding, huge file |
| **v2 (Current)** | Sentence Transformers + ChromaDB + Gemini LLM | ~50 MB ChromaDB | Semantic search, mood-aware, conversational AI |

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgements

- [TMDB](https://www.themoviedb.org/) for the movie database API
- [MovieLens](https://grouplens.org/datasets/movielens/) for the ratings dataset
- [Sentence Transformers](https://www.sbert.net/) for the embedding model
- [ChromaDB](https://www.trychroma.com/) for the vector database
- [Google Gemini](https://ai.google.dev/) for the LLM capabilities
