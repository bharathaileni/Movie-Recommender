🎬 Cross-Cultural Movie Recommender System
A content-based movie recommendation engine that bridges the gap between Hollywood and Indian Cinema (Bollywood, Tollywood, etc.). This system analyzes over 16,000 movies to provide seamless recommendations based on plot, genre, cast, and crew.

🚀 Features
Hybrid Dataset: Combined MovieLens and TMDB Hollywood data with custom Indian movie datasets.

Content-Based Filtering: Uses Natural Language Processing (NLP) to vectorize movie metadata.

Smart Search: Fuzzy matching logic to find movies even with slightly different titles or years.

Dynamic Posters: Real-time poster fetching using the TMDB API for a professional UI experience.

Interactive UI: A clean, fast web interface built with Streamlit.

🛠️ Tech Stack
Language: Python 3.x

Frontend: Streamlit

Data Analysis: Pandas, NumPy

Machine Learning: Scikit-Learn (CountVectorizer, Cosine Similarity)

API: The Movie Database (TMDB)

📦 Installation & Setup
Clone the repository

Bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
Create a Virtual Environment

Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies

Bash
pip install -r requirements.txt
Generate the Models
Run the main processing script to clean the data and create the similarity matrix:

Bash
python main.py
Set up API Key

Get an API key from The Movie Database (TMDB).

Open app.py and replace YOUR_TMDB_API_KEY with your actual key.

Run the App

Bash
streamlit run app.py
🧠 How it Works
The system creates a "tag" for every movie by combining the plot summary, genres, top 3 actors, and the director. These tags are processed using Bag of Words (CountVectorizer) to convert text into vectors. We then calculate the Cosine Similarity between these vectors to find the closest matches for any given movie.

📊 Dataset Sources
MovieLens 20M Dataset

TMDB 5000 Movies

Indian Movies Metadata (IMDB & custom scraped data)

🤝 Contributing
Feel free to fork this project, open issues, or submit pull requests to add support for more regional languages or improve the recommendation algorithm!
