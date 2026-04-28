"""
app.py — Upgraded Movie Recommender with Dual Mode
===================================================
🎯 Classic Mode: Pick a movie → get 5 similar ones (original behavior)
💬 Mood Mode:    Describe your mood → LLM interprets → smart recommendations

Powered by: ChromaDB + Sentence Transformers + Google Gemini
"""

import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
import chromadb
from config import (
    TMDB_API_KEY, EMBEDDING_MODEL,
    CHROMA_DB_PATH, CHROMA_COLLECTION,
    TOP_N_CANDIDATES, TOP_N_DISPLAY
)
from llm_service import interpret_mood_and_recommend

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Movie Matcher — AI Powered",
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown("""
<style>
    /* Main header gradient */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    /* Movie card styling */
    .movie-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        border: 1px solid #333;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    /* LLM response box */
    .llm-response {
        background: linear-gradient(145deg, #0f0c29, #302b63, #24243e);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #444;
        color: #e0e0e0;
        line-height: 1.7;
        margin: 20px 0;
    }
    /* Mode selector tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 24px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD RESOURCES (cached for performance)
# ============================================================
@st.cache_resource
def load_chromadb():
    """Load ChromaDB client and collection."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=CHROMA_COLLECTION)
    return collection

@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)

# Initialize resources
try:
    collection = load_chromadb()
    embed_model = load_embedding_model()
    db_ready = True
except Exception as e:
    db_ready = False
    db_error = str(e)

# ============================================================
# POSTER FETCHING (preserved from original)
# ============================================================
@st.cache_data(ttl=86400)  # Cache posters for 24 hours
def fetch_poster(movie_id: int, movie_title: str) -> str:
    """Fetch movie poster from TMDB API. Supports both ID and title search."""
    if movie_id != 0:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    else:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"

    try:
        response = requests.get(url, timeout=5)
        data = response.json()

        if movie_id == 0:
            poster_path = data['results'][0]['poster_path']
        else:
            poster_path = data['poster_path']

        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"

# ============================================================
# RECOMMENDATION ENGINES
# ============================================================
def recommend_by_movie(movie_title: str) -> tuple[list[dict], list[str]]:
    """
    Classic Mode: Find movies similar to a given movie title.
    Encodes the movie's tags with the same model, then queries ChromaDB.
    """
    # Use ChromaDB's where filter to find the movie efficiently (no full scan)
    match = collection.get(
        where={"title": movie_title},
        include=["embeddings", "metadatas"]
    )

    if not match['ids']:
        return [], []

    # Get the movie's embedding and query for similar ones
    target_embedding = match['embeddings'][0]
    results = collection.query(
        query_embeddings=[target_embedding],
        n_results=TOP_N_DISPLAY + 1,  # +1 because the movie itself will be in results
        include=["metadatas", "distances"]
    )

    movies = []
    posters = []
    for meta in results['metadatas'][0]:
        if meta['title'] != movie_title:  # Skip the query movie itself
            movies.append(meta)
            posters.append(fetch_poster(meta['tmdbId'], meta['title']))
        if len(movies) >= TOP_N_DISPLAY:
            break

    return movies, posters


def recommend_by_mood(mood_text: str) -> tuple[list[dict], list[str], dict]:
    """
    Mood Mode: Encode the mood text, find candidates in ChromaDB,
    then use Gemini LLM to re-rank and generate a conversational response.
    Falls back to raw ChromaDB results if the LLM call fails.
    """
    # Encode the mood text into the same 384-dim space
    mood_embedding = embed_model.encode(mood_text).tolist()

    # Query ChromaDB for the top N candidates
    results = collection.query(
        query_embeddings=[mood_embedding],
        n_results=TOP_N_CANDIDATES,
        include=["metadatas", "distances"]
    )

    candidates = results['metadatas'][0]

    # Try to send candidates to Gemini for re-ranking
    try:
        llm_result = interpret_mood_and_recommend(mood_text, candidates)
    except Exception as e:
        # Graceful fallback: use raw ChromaDB results without LLM
        llm_result = {
            "response": None,
            "recommended_titles": [m['title'] for m in candidates[:TOP_N_DISPLAY]],
            "error": str(e)
        }

    # Match the LLM's recommended titles back to our metadata for posters
    recommended_titles = llm_result['recommended_titles']
    final_movies = []
    posters = []

    # Build a lookup from candidate list
    candidate_lookup = {m['title']: m for m in candidates}

    for title in recommended_titles:
        if title in candidate_lookup:
            meta = candidate_lookup[title]
            final_movies.append(meta)
            posters.append(fetch_poster(meta['tmdbId'], meta['title']))
        if len(final_movies) >= TOP_N_DISPLAY:
            break

    # If LLM didn't return enough, fill from candidates
    if len(final_movies) < TOP_N_DISPLAY:
        for meta in candidates:
            if meta['title'] not in [m['title'] for m in final_movies]:
                final_movies.append(meta)
                posters.append(fetch_poster(meta['tmdbId'], meta['title']))
            if len(final_movies) >= TOP_N_DISPLAY:
                break

    return final_movies, posters, llm_result

# ============================================================
# UI
# ============================================================
st.markdown('<p class="main-header">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Discover movies from Hollywood, Tollywood, and beyond — now with AI-powered mood matching!</p>', unsafe_allow_html=True)

if not db_ready:
    st.error(f"⚠️ ChromaDB not found. Please run `python build_vectordb.py` first.\n\nError: {db_error}")
    st.stop()

# Show database stats
movie_count = collection.count()
st.caption(f"🎞️ Database: **{movie_count:,}** movies indexed in ChromaDB")

# --- Dual Mode Tabs ---
tab_classic, tab_mood = st.tabs(["🎯 Classic Mode", "💬 Mood Mode"])

# ============================================================
# CLASSIC MODE TAB
# ============================================================
with tab_classic:
    st.markdown("**Pick a movie you liked, and we'll find similar ones.**")

    # Get all movie titles for the dropdown
    @st.cache_data
    def get_all_titles():
        all_data = collection.get(include=["metadatas"])
        return sorted([m['title'] for m in all_data['metadatas']])

    all_titles = get_all_titles()

    selected_movie = st.selectbox(
        "Search for a movie:",
        all_titles,
        index=None,
        placeholder="Type to search...",
        key="classic_select"
    )

    if st.button("✨ Show Recommendations", key="classic_btn", type="primary"):
        if selected_movie:
            with st.spinner("🔍 Finding similar movies..."):
                movies_result, posters_result = recommend_by_movie(selected_movie)

                if movies_result:
                    st.subheader(f'Since you liked "{selected_movie}", you might enjoy:')
                    cols = st.columns(TOP_N_DISPLAY)
                    for idx, col in enumerate(cols):
                        if idx < len(movies_result):
                            with col:
                                st.image(posters_result[idx], width="stretch")
                                st.markdown(f"**{movies_result[idx]['title']}**")
                else:
                    st.error("Could not find recommendations for this movie.")
        else:
            st.warning("Please select a movie first.")

# ============================================================
# MOOD MODE TAB
# ============================================================
with tab_mood:
    st.markdown("**Describe your mood, and our AI will find the perfect movies for you.**")

    mood_input = st.text_area(
        "How are you feeling? What kind of movie are you in the mood for?",
        placeholder="e.g., I'm feeling nostalgic and want something heartwarming from the 90s...\n"
                    "or: I need an intense thriller that keeps me on the edge of my seat...\n"
                    "or: Something fun and colorful for a family movie night...",
        height=120,
        key="mood_input"
    )

    col_btn, col_examples = st.columns([1, 3])
    with col_btn:
        mood_submitted = st.button("🤖 Get AI Recommendations", key="mood_btn", type="primary")

    with col_examples:
        st.markdown(
            "<small style='color:#888'>💡 Try: <em>\"sad romantic drama\"</em> · "
            "<em>\"action-packed superhero adventure\"</em> · "
            "<em>\"feel-good comedy to watch with friends\"</em></small>",
            unsafe_allow_html=True
        )

    if mood_submitted:
        if mood_input.strip():
            with st.spinner("🧠 AI is analyzing your mood and searching our database..."):
                movies_result, posters_result, llm_result = recommend_by_mood(mood_input)

                # Check if LLM worked or we're in fallback mode
                if llm_result.get("error"):
                    st.warning(
                        "⚠️ The AI assistant is temporarily unavailable (API quota exceeded). "
                        "Showing the best semantic matches from our database instead!"
                    )
                elif llm_result.get("response"):
                    # Display the LLM's conversational response
                    st.markdown(
                        f'<div class="llm-response">{llm_result["response"]}</div>',
                        unsafe_allow_html=True
                    )

                # Display movie cards
                if movies_result:
                    st.subheader("🎬 Here are your picks:")
                    cols = st.columns(TOP_N_DISPLAY)
                    for idx, col in enumerate(cols):
                        if idx < len(movies_result):
                            with col:
                                st.image(posters_result[idx], width="stretch")
                                st.markdown(f"**{movies_result[idx]['title']}**")
        else:
            st.warning("Please describe your mood first!")

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>"
    "Powered by <strong>ChromaDB</strong> · <strong>Sentence Transformers</strong> · "
    "<strong>Google Gemini</strong> · <strong>TMDB API</strong>"
    "</div>",
    unsafe_allow_html=True
)