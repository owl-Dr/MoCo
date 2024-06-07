import streamlit as st
import requests
import pickle
import numpy as np


# Load the necessary dictionaries and models
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)


models_and_data = load_pickle('models_and_data.pkl')
id_to_title = models_and_data['id_to_title']
title_to_id = models_and_data['title_to_id']
id_to_genres = models_and_data['id_to_genres']
id_to_overview = models_and_data['id_to_overview']
id_to_cast = models_and_data['id_to_cast']
reduced_features = models_and_data['reduced_features']
knn_model = models_and_data['knn_model']
rf_model = models_and_data['rf_model']


# Function to fetch movie details from TMDB API
def fetch_movie_details(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        details = {
            "poster_path": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path', ''),
            "title": data.get('title', 'N/A'),
            "genres": ', '.join([genre['name'] for genre in data.get('genres', [])]),
            "popularity": data.get('popularity', 'N/A'),
            "overview": data.get('overview', 'N/A')
        }
        return details
    except Exception as e:
        st.error(f"Error fetching details for movie ID {movie_id}: {e}")
        return None


# Function to get movie recommendations
def recommend(movie_title):
    if movie_title not in title_to_id:
        st.error(f"Movie '{movie_title}' not found in the dataset.")
        return []

    movie_id = title_to_id[movie_title]
    movie_index = list(id_to_title.keys()).index(movie_id)

    # Find nearest neighbors using KNN
    distances, indices = knn_model.kneighbors([reduced_features[movie_index]], n_neighbors=6)
    knn_indices = indices.flatten()[1:]  # Get top 5 excluding the movie itself

    # Get Random Forest predictions
    rf_distances = rf_model.predict([reduced_features[movie_index]])
    rf_indices = np.argsort(rf_distances)[:6]  # Get top 5 excluding the movie itself

    # Combine results and exclude the selected movie
    combined_indices = list(set(knn_indices).union(set(rf_indices)))
    combined_indices = [idx for idx in combined_indices if idx != movie_index]

    # Get the recommended movie details
    recommended_movie_details = []
    for idx in combined_indices:
        recommended_movie_id = list(id_to_title.keys())[idx]
        movie_details = fetch_movie_details(recommended_movie_id)
        if movie_details:
            recommended_movie_details.append(movie_details)

    return recommended_movie_details


# Streamlit UI
st.set_page_config(page_title='MoCo: Movie Companion', page_icon=':clapper:')
st.title('MoCo: Movie Companion')
st.subheader('Discover Your Next Favorite Movie')

# Get movie list
movie_list = list(id_to_title.values())
selected_movie = st.selectbox(
    "Search for a movie by typing or selecting from the dropdown",
    movie_list
)

if st.button('Get Recommendations'):
    recommended_movies = recommend(selected_movie)
    if recommended_movies:
        st.session_state['selected_movie'] = selected_movie
        st.session_state['recommended_movies'] = recommended_movies

# Display recommendations in rows with poster on the left and details on the right
if 'recommended_movies' in st.session_state:
    recommended_movies = st.session_state['recommended_movies']
    st.markdown(f"### Recommended Movies Based on '{st.session_state['selected_movie']}'")
    for movie in recommended_movies:
        st.markdown("---")
        cols = st.columns([1, 3])
        with cols[0]:
            st.image(movie['poster_path'], width=150)
        with cols[1]:
            st.markdown(f"### {movie['title']}")
            st.write(f"**Genres:** {movie['genres']}")
            st.write(f"**Popularity:** {movie['popularity']}")
            st.write(f"**Overview:** {movie['overview']}")
            add_watchlist = st.checkbox(f'Add "{movie["title"]}" to Watchlist', key=f'add_watchlist_{movie["title"]}')
            if add_watchlist:
                if 'watchlist' not in st.session_state:
                    st.session_state['watchlist'] = []
                if movie['title'] not in st.session_state['watchlist']:
                    st.session_state['watchlist'].append(movie['title'])

# Display watchlist
st.sidebar.header("Your Watchlist")
if 'watchlist' in st.session_state:
    for movie_title in st.session_state['watchlist']:
        st.sidebar.write(movie_title)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<center>MoCo may not work in India; please use a VPN if necessary.</center>", unsafe_allow_html=True)
