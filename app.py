import streamlit as st
import pickle
import pandas as pd
import requests

# Set page title and icon
st.set_page_config(page_title="Movie Matcher", page_icon="🎬", layout="wide")

# 1. Load Data with Caching
@st.cache_data
def load_data():
    movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return movies, similarity

movies, similarity = load_data()

# 2. Smart Poster Fetching (Supports Hollywood & Indian Search)
@st.cache_data
def fetch_poster(movie_id, movie_title):
    api_key = "YOUR_TMDB_API_KEY" # <--- REPLACE THIS WITH YOUR KEY
    
    # If we have a TMDB ID, use the direct lookup
    if movie_id != 0:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=1416d4d6db3ba8643e61734e3ac8c41a&language=en-US"
    # Otherwise, search by title (for your Indian movies)
    else:
        url = f"https://api.themoviedb.org/3/search/movie?api_key=1416d4d6db3ba8643e61734e3ac8c41a&query={movie_title}"

    try:
        response = requests.get(url)
        data = response.json()
        
        if movie_id == 0:
            poster_path = data['results'][0]['poster_path']
        else:
            poster_path = data['poster_path']
            
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"

# 3. Recommendation Engine
def recommend(movie):
    try:
        movie_index = movies[movies['title'] == movie].index[0]
        distances = similarity[movie_index]
        # Get top 5 matches
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        recs = []
        posters = []
        for i in movies_list:
            m_id = movies.iloc[i[0]].tmdbId
            m_title = movies.iloc[i[0]].title
            recs.append(m_title)
            posters.append(fetch_poster(m_id, m_title))
        return recs, posters
    except:
        return ["Error finding recommendations"], []

# 4. Streamlit UI Design
st.title('🎬 Movie Recommender System')
st.markdown("Discover movies from Hollywood, Tollywood, and beyond!")

selected_movie = st.selectbox(
    'Search for a movie you liked:',
    movies['title'].values
)

if st.button('✨ Show Recommendations'):
    with st.spinner('Checking our cinematic database...'):
        names, posters = recommend(selected_movie)
        
        if names:
            st.subheader(f'Since you liked "{selected_movie}", you might enjoy:')
            cols = st.columns(5)
            for idx, col in enumerate(cols):
                with col:
                    st.image(posters[idx])
                    st.write(f"**{names[idx]}**")
        else:
            st.error("Could not find recommendations for this movie.")

# Footer
st.markdown("---")
st.caption("Powered by Scikit-Learn, Pandas, and TMDB API")