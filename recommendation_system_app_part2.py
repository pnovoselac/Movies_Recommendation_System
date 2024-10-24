import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from surprise import Reader, Dataset, accuracy
import requests
from PIL import Image
import os
from demographic_filtering import q_movies

TMDB_API_KEY = "0637182b533e3b79ca4aa283f9630e38"
# Učitavanje podataka
@st.cache_data
def load_data():
    movies_dataset = pd.read_csv('new_movies_dataset.csv')
    movies_dataset['bag_of_words'] = movies_dataset['bag_of_words'].astype(str).fillna('')
    ratings_df = pd.read_csv('new_ratings_dataset.csv')
    weighted_df = pd.read_csv('weighted_df.csv', index_col=0)
    liked_movies_df = pd.read_csv('liked_movies.csv')
    return movies_dataset, ratings_df, weighted_df, liked_movies_df

movies_dataset, ratings_df, weighted_df, liked_movies_df = load_data()

# Učitavanje modela
@st.cache_data
def load_model():
    with open('best_model_fit.pkl', 'rb') as f:
        algo, trainset = pickle.load(f)
    algo.trainset = trainset
    return algo

algo = load_model()


# Funkcija za preporuke pomoću filtriranja temeljenog na sadržaju
def get_recommendations(title, model, indices):
    idx = indices[title]
    distances, indices = model.kneighbors(count_matrix[idx], n_neighbors=11)
    indices = indices.flatten()[1:]
    return movies_dataset['title'].iloc[indices], movies_dataset['id'].iloc[indices]

# Funkcija za dohvaćanje plakata filma
def get_poster_path(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return "https://image.tmdb.org/t/p/w500/" + data['poster_path'] if data['poster_path'] else None
        return None
    except Exception as e:
        st.error(f"Error fetching poster path for movie ID {movie_id}: {e}")
        return None

# Funkcija za izračun ponderirane ocjene
def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

# Funkcija za prikaz filmova u jednom redu po 5
def display_movies_in_row(movies, num_cols=5):
    for i in range(0, len(movies), num_cols):
        row_movies = movies[i:i + num_cols]
        cols = st.columns(len(row_movies))
        for col, movie in zip(cols, row_movies):
            with col:
                st.image(movie['poster_path'], width=150)
                st.write(movie['title'])
                st.write(f"Score: {movie['score']:.2f}" if movie['score'] is not None else "")

# Funkcija za pronalazak ID-a filma na temelju naziva
def get_movie_id(movie_title, movies_dataset):
    result = movies_dataset[movies_dataset['title'].str.lower() == movie_title.lower()]
    if not result.empty:
        return result.iloc[0]['id']
    return None

# Funkcija za spremanje lajkanih filmova u CSV datoteku
def save_liked_movies(liked_movies):
    df = pd.DataFrame(liked_movies, columns=['title'])
    if not os.path.isfile('liked_movies.csv'):
        df.to_csv('liked_movies.csv', index=False)
    else:  # else it exists so append without writing the header
        df.to_csv('liked_movies.csv', mode='a', header=False, index=False)

# Streamlit sučelje
st.title("Movies Recommendation System")
st.sidebar.title("Select Algorithm")

# Sidebar za odabir algoritma
algorithm = st.sidebar.selectbox("Choose an algorithm", ("Demographic Filtering", "Collaborative Filtering", "Content-Based Filtering"))



if algorithm == "Collaborative Filtering":
    user_id = st.number_input("Enter User ID", min_value=1, max_value=int(ratings_df['userId'].max()))
    movie_title = st.text_input("Enter Movie Title")
    
    if st.button("Get Recommendation"):
        movie_id = get_movie_id(movie_title, movies_dataset)
        
        if movie_id is not None:
            prediction = algo.predict(user_id, movie_id).est
            st.write(f"The predicted rating for user ID {user_id} and movie ID {movie_id} ({movie_title}) is: {prediction}")
            
            poster_path = get_poster_path(movie_id)
            if poster_path:
                st.image(poster_path, caption=f"Poster for Movie: {movie_title}")
        else:
            st.write(f"Movie titled '{movie_title}' not found.")

elif algorithm == "Content-Based Filtering":
    movie_title = st.selectbox("Choose a movie", movies_dataset['title'].values)
    if st.button("Get Recommendations"):
        count = CountVectorizer(stop_words='english')
        count_matrix = count.fit_transform(movies_dataset['bag_of_words'])
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(count_matrix)
        indices = pd.Series(movies_dataset.index, index=movies_dataset['title'])

        recommended_movies, movie_ids = get_recommendations(movie_title, model, indices)

        st.session_state.recommended_movies = recommended_movies
        st.session_state.movie_ids = movie_ids
        st.session_state.liked_movies = []

    if 'recommended_movies' in st.session_state:
        st.write("Top 10 movie recommendations based on content filtering:")
        movies = []
        for title, movie_id in zip(st.session_state.recommended_movies, st.session_state.movie_ids):
            poster_path = get_poster_path(movie_id)
            movies.append({'title': title, 'poster_path': poster_path, 'score': None})
        display_movies_in_row(movies)

        # Prikazivanje filmova i omogućavanje korisniku da odabere one koji mu se sviđaju
        for movie in movies:
            col1, col2 = st.columns([1, 5])
            with col1:
                st.image(movie['poster_path'], width=150)
            with col2:
                if st.checkbox(movie['title'], key=movie['title']):
                    if movie['title'] not in st.session_state.liked_movies:
                        st.session_state.liked_movies.append(movie['title'])
                else:
                    if movie['title'] in st.session_state.liked_movies:
                        st.session_state.liked_movies.remove(movie['title'])

        # Izračun preciznosti
        k = 10

          # Spremanje lajkanih filmova u CSV datoteku
        if st.button("Save Liked Movies"):
            save_liked_movies(st.session_state.liked_movies)
            st.success("Liked movies saved successfully.")

        new_liked_movies_df = pd.read_csv('liked_movies.csv')

        num_liked_movies = len(st.session_state.liked_movies)
        precision_at_k = num_liked_movies / k
        recall_at_k = num_liked_movies / new_liked_movies_df.shape[0]

        
        st.write(f"Precision at K: {precision_at_k:.4f}")
        st.write(f"Recall at K: {recall_at_k:.4f}")

        

elif algorithm == "Demographic Filtering":
    st.write("### Top Movies by Demographic Filtering")
    movies = []
    for idx, row in q_movies.head(15).iterrows():
        poster_path = get_poster_path(row['id'])
        movies.append({'title': row['title'], 'poster_path': poster_path, 'score': row['score']})
    display_movies_in_row(movies)
