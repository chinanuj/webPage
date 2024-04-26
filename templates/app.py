from flask import Flask, render_template, request, jsonify
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np
from pprint import pprint
app = Flask(__name__,template_folder='templates')
column_names1 = ['userId','movieId','rating','timestamp']
dataset = pd.read_csv("ratings.csv")
movies = pd.read_csv('movies.csv')
movie_dataset = movies[['movieId','title']]
movie_titles = movies["title"].tolist()
merged_dataset = pd.merge(dataset, movie_dataset, how='inner', on='movieId')
refined_dataset = merged_dataset.groupby(by=['userId','title'], as_index=False).agg({"rating":"mean"})
num_users = len(refined_dataset['userId'].value_counts())
num_items = len(refined_dataset['title'].value_counts())
rating_count_df = pd.DataFrame(refined_dataset.groupby(['rating']).size(), columns=['count'])
total_count = num_items * num_users
zero_count = total_count-refined_dataset.shape[0]
# Append using pd.concat instead of DataFrame.append
rating_count_df = pd.concat(
    [rating_count_df, pd.DataFrame({'count': zero_count}, index=[0.0])],
    ignore_index=False
).sort_index()
rating_count_df['log_count'] = np.log(rating_count_df['count'])
rating_count_df = rating_count_df.reset_index().rename(columns={'index': 'rating score'})
# get rating frequency
movies_count_df = pd.DataFrame(refined_dataset.groupby('title').size(), columns=['count'])

# pivot and create movie-user matrix
user_to_movie_df = refined_dataset.pivot(
    index='userId',
     columns='title',
      values='rating').fillna(0)

# transform matrix to scipy sparse matrix
user_to_movie_sparse_df = csr_matrix(user_to_movie_df.values)
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_to_movie_sparse_df)

def recommender_system(user_id, n_similar_users=5, n_movies=10):
    def get_similar_users(user, n=5):
        knn_input = np.asarray([user_to_movie_df.values[user-1]])
        distances, indices = knn_model.kneighbors(knn_input, n_neighbors=n+1)
        return indices.flatten()[1:] + 1, distances.flatten()[1:]

    def filtered_movie_recommendations(n=10):
        movies = pd.read_csv('movies.csv')
        links_df = pd.read_csv('links.csv')
        first_zero_index = np.where(mean_rating_list == 0)[0][-1]
        sortd_index = np.argsort(mean_rating_list)[::-1]
        sortd_index = sortd_index[:list(sortd_index).index(first_zero_index)]
        n = min(len(sortd_index), n)
        movies_watched = list(refined_dataset[refined_dataset['userId'] == user_id]['title'])
        filtered_movie_list = list(movies_list[sortd_index])
        count = 0
        final_movie_list = []
        for i in filtered_movie_list:
            if i not in movies_watched:
                count += 1
                final_movie_list.append(i)
            if count == n:
                break
        for i in range(len(final_movie_list)):
            movie_id = movies.loc[movies['title'] == final_movie_list[i], 'movieId'].tolist()[0]
            movie_id = links_df.loc[links_df['movieId'] == movie_id, 'tmdbId'].tolist()[0]
            final_movie_list[i] = [final_movie_list[i],int(movie_id)]

        return final_movie_list

    similar_user_list, distance_list = get_similar_users(user_id, n_similar_users)
    weightage_list = distance_list / np.sum(distance_list)
    mov_rtngs_sim_users = user_to_movie_df.values[similar_user_list]
    movies_list = user_to_movie_df.columns
    weightage_list = weightage_list[:, np.newaxis] + np.zeros(len(movies_list))
    new_rating_matrix = weightage_list * mov_rtngs_sim_users
    mean_rating_list = new_rating_matrix.sum(axis=0)
    return filtered_movie_recommendations(n_movies)

# pivot and create movie-user matrix
movie_to_user_df = refined_dataset.pivot(
     index='title',
   columns='userId',
      values='rating').fillna(0)


# transform matrix to scipy sparse matrix
movie_to_user_sparse_df = csr_matrix(movie_to_user_df.values)
movies_list = list(movie_to_user_df.index)
movie_dict = {movie : index for index, movie in enumerate(movies_list)}
knn_movie_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_movie_model.fit(movie_to_user_sparse_df)
## function to find top n similar users of the given input user


def recommend_movies(movie, n=10):
    movies = pd.read_csv('movies.csv')
    links_df = pd.read_csv('links.csv')

    index = movie_dict[movie]
    knn_input = np.asarray([movie_to_user_df.values[index]])
    n = min(len(movies_list) - 1, n)
    distances, indices = knn_movie_model.kneighbors(knn_input, n_neighbors=n + 1)

    recommended_movies = []
    for i in range(1, len(distances[0])):
        movie_id = movies.loc[movies['title'] == movies_list[indices[0][i]], 'movieId'].tolist()[0]
        movie_id = links_df.loc[links_df['movieId'] == movie_id, 'tmdbId'].tolist()[0]
        recommended_movies.append([movies_list[indices[0][i]], movie_id])
    return recommended_movies


import pandas as pd
import requests

# Read movies.csv and links.csv files
movies = pd.read_csv('movies.csv')
links_df = pd.read_csv('links.csv')

# Merge movies_df and links_df on 'movieId' column to get TMDb IDs for each movie
merged_df = pd.merge(movies, links_df, on='movieId')

# Create a dictionary to store movie titles and their corresponding TMDb IDs
movie_tmdb_mapping = {}

# Iterate over merged dataframe rows to populate the dictionary
for index, row in merged_df.iterrows():
    movie_title = row['title']
    tmdb_id = row['tmdbId']

    # Append TMDb ID to the list of TMDb IDs for the movie title
    if movie_title not in movie_tmdb_mapping:
        movie_tmdb_mapping[movie_title] = [tmdb_id]
    else:
        movie_tmdb_mapping[movie_title].append(tmdb_id)

# Convert the dictionary values to sets to remove duplicates
movie_tmdb_mapping = {title: list(set(ids)) for title, ids in movie_tmdb_mapping.items()}

# Function to fetch movie poster URL from TMDb API
def get_movie_poster_url(tmdb_id):
    # TMDb API base URL and API key (replace 'YOUR_API_KEY' with your actual API key)
    base_url = "https://api.themoviedb.org/3"
    api_key = "cebff0ded35e48388de25551bebeec40"

    # Endpoint for searching movies by ID
    endpoint = f"{base_url}/movie/{tmdb_id}"

    # Parameters for the request (including API key)
    params = {
        'api_key': api_key
    }

    try:
        # Send GET request to TMDb API
        response = requests.get(endpoint, params=params)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes

        # Parse JSON response
        movie_data = response.json()

        # Extract poster URL from the response
        poster_path = movie_data.get('poster_path')

        # Construct full poster URL
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return poster_url
        else:
            return None

    except requests.exceptions.RequestException as e:
        print("Error fetching movie poster:", e)
        return None


@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    return jsonify(movie_titles)

@app.route('/recommend_movies', methods=['POST'])
def get_recommendations():
    movie_name = request.form['movie_name']
    recommended_movies = recommend_movies(movie_name)
    return jsonify(recommended_movies)


@app.route('/user_recommendations', methods=['POST'])
def user_recommendations():
    user_id = int(request.form['user_id'])
    # Call the recommender_system function
    recommended_movies = recommender_system(user_id, n_similar_users=5, n_movies=10)
    # Return the recommended movies as a dictionary
    return jsonify({"movies": recommended_movies})




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_name = request.form['movie_name']

    # Get recommendations for the given movie
    recommended_movies = recommend_movies(movie_name)

    return render_template('recommendations.html', movies=recommended_movies)


if __name__ == '__main__':
    app.run(debug=True)
