import pandas as pd
import ast
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- STEP 1: LOAD & MERGE HOLLYWOOD DATA ---
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

# --- STEP 2: CLEANING FUNCTIONS ---
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

# FIXED: Defensive collapse function to handle NaNs/Floats
def collapse(L):
    L1 = []
    for i in L:
        if isinstance(i, str) and i.lower() != 'nan':
            L1.append(i.replace(" ",""))
    return L1

# --- STEP 3: APPLY CLEANING TO HOLLYWOOD ---
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

# --- STEP 4: LOAD & CLEAN INDIAN DATA ---
indian_meta = pd.read_csv('data/movies_data.csv')
indian_desc = pd.read_csv('data/IMDB_10000.csv')

# Use OUTER join to keep ALL movies from both files
indian_full = indian_meta.merge(indian_desc, left_on='Name', right_on='title', how='outer')

# FIXED: Fill NaNs so the logic doesn't crash
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

# --- STEP 5: COMBINE EVERYTHING ---
new_df = pd.concat([hw_final, indian_final], ignore_index=True)

# Convert tags from lists back to strings
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

print(f"Total Combined Database: {len(new_df)} movies ready!")

# --- STEP 6: THE MACHINE LEARNING MATH ---
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vector)

# --- STEP 7: RECOMMENDATION FUNCTION ---
def recommend(movie_title):
    search_term = movie_title.strip().lower()
    match = new_df[new_df['title'].str.lower().str.contains(search_term, na=False)]
    
    if not match.empty:
        match = match.iloc[match['title'].str.len().argsort()]
        movie_index = match.index[0]
        actual_title = match.iloc[0]['title']
        
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"\nRecommendations for '{actual_title}':")
        for i in movies_list:
            print(f"- {new_df.iloc[i[0]].title}")
    else:
        print(f"Movie '{movie_title}' not found in database.")

# --- STEP 8: SAVE FOR WEB APP ---
pickle.dump(new_df.to_dict(), open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Models saved successfully!")

# TEST CALLS
recommend('Toy Story')
recommend('Kantara')

