
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv('data/ml-1m/ratings.dat', sep='::', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
movies = pd.read_csv('data/ml-1m/movies.dat', sep='::', 
                     names=['movie_id', 'title', 'genres'], engine='python', encoding='latin-1')

# Create genre features
all_genres = set()
for genres in movies['genres']:
    all_genres.update(genres.split('|'))
for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)

# Calculate popularity scores
movie_stats = ratings.groupby('movie_id').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
).reset_index()
movie_stats = movie_stats.merge(movies[['movie_id', 'title', 'genres']], on='movie_id')
C = movie_stats['avg_rating'].mean()
m = movie_stats['num_ratings'].quantile(0.75)
movie_stats['score'] = movie_stats.apply(
    lambda x: (x['num_ratings']/(x['num_ratings']+m) * x['avg_rating']) + (m/(x['num_ratings']+m) * C), axis=1)

# Genre similarity
genre_matrix = movies[list(all_genres)].values
genre_similarity = cosine_similarity(genre_matrix)

# Initialize app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("MovieLens Recommender System", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Hr(),

    dcc.Tabs([
        dcc.Tab(label='Popular Movies', children=[
            html.Div([
                html.H3("Top Rated Movies"),
                html.Label("Select Genre:"),
                dcc.Dropdown(
                    id='genre-dropdown',
                    options=[{'label': 'All Genres', 'value': 'all'}] + 
                            [{'label': g, 'value': g} for g in sorted(all_genres)],
                    value='all',
                    style={'width': '300px'}
                ),
                html.Br(),
                html.Div(id='popular-movies-output')
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Find Similar Movies', children=[
            html.Div([
                html.H3("Content-Based Recommendations"),
                html.Label("Enter a movie title:"),
                dcc.Dropdown(
                    id='movie-dropdown',
                    options=[{'label': t, 'value': i} for i, t in zip(movies['movie_id'], movies['title'])],
                    placeholder="Search for a movie...",
                    style={'width': '400px'}
                ),
                html.Br(),
                html.Div(id='similar-movies-output')
            ], style={'padding': '20px'})
        ]),

        dcc.Tab(label='Dataset Stats', children=[
            html.Div([
                html.H3("Dataset Statistics"),
                html.P(f"Total Ratings: {len(ratings):,}"),
                html.P(f"Total Movies: {len(movies):,}"),
                html.P(f"Total Users: {ratings['user_id'].nunique():,}"),
                html.P(f"Rating Range: 1-5 stars"),
                html.P(f"Genres: {len(all_genres)}"),
                html.Hr(),
                html.H4("Genre Distribution"),
                html.P(", ".join(sorted(all_genres)))
            ], style={'padding': '20px'})
        ])
    ])
])

@app.callback(
    Output('popular-movies-output', 'children'),
    Input('genre-dropdown', 'value')
)
def update_popular(genre):
    if genre == 'all':
        df = movie_stats.nlargest(15, 'score')
    else:
        genre_movies = movies[movies[genre] == 1]['movie_id']
        df = movie_stats[movie_stats['movie_id'].isin(genre_movies)].nlargest(15, 'score')

    return html.Table([
        html.Tr([html.Th("Rank"), html.Th("Title"), html.Th("Rating"), html.Th("Votes"), html.Th("Genres")])
    ] + [
        html.Tr([
            html.Td(i+1), 
            html.Td(row['title']), 
            html.Td(f"{row['avg_rating']:.2f}"),
            html.Td(f"{row['num_ratings']:,}"),
            html.Td(row['genres'])
        ]) for i, (_, row) in enumerate(df.iterrows())
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

@app.callback(
    Output('similar-movies-output', 'children'),
    Input('movie-dropdown', 'value')
)
def update_similar(movie_id):
    if movie_id is None:
        return "Select a movie to see recommendations"

    idx = movies[movies['movie_id'] == movie_id].index[0]
    sim_scores = list(enumerate(genre_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    movie_name = movies.iloc[idx]['title']
    movie_genres = movies.iloc[idx]['genres']

    results = [html.H4(f"Movies similar to: {movie_name}"), html.P(f"Genres: {movie_genres}"), html.Hr()]

    for i, (mid, score) in enumerate(sim_scores):
        title = movies.iloc[mid]['title']
        genres = movies.iloc[mid]['genres']
        results.append(html.P(f"{i+1}. {title} ({genres}) - Similarity: {score:.2f}"))

    return html.Div(results)

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
