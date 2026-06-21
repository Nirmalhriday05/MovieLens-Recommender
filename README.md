# MovieLens Recommender System

## Project Overview
A comprehensive movie recommendation system built using the MovieLens 1M dataset, implementing multiple recommendation algorithms.

## Dataset
- **Source**: MovieLens 1M Dataset
- **Size**: 1,000,209 ratings from 6,040 users on 3,883 movies
- **Rating Scale**: 1-5 stars

## Models Implemented

### 1. Popularity-Based Recommender
- Uses weighted rating formula (similar to IMDB)
- Balances average rating with number of votes
- Best for new users (cold start problem)

### 2. User-Based Collaborative Filtering
- Finds similar users based on rating patterns
- Recommends movies liked by similar users
- Uses cosine similarity

### 3. Item-Based Collaborative Filtering
- Finds similar movies based on user ratings
- More stable than user-based approach
- Uses cosine similarity

### 4. Matrix Factorization (SVD)
- Discovers latent factors in user-movie interactions
- 50 latent factors used
- Handles sparsity well

### 5. Content-Based Filtering
- Recommends movies based on genre similarity
- Uses one-hot encoding for 18 genres
- Good for new movies (no ratings needed)

### 6. Hybrid Recommender
- Combines Popularity, Content-Based, and Collaborative Filtering scores into one weighted recommendation
- Balances new-user coverage (popularity) with personalization (content + collaborative)
- The most complete model in the project, built to overcome each individual method's weak points

## Evaluation Metrics
| Model | RMSE | MAE |
|-------|------|-----|
| Global Mean | 1.1131 | 0.9304 |
| User Mean | 1.0320 | 0.8264 |
| Movie Mean | 0.9764 | 0.7804 |

## Project Structure
```
MovieLens-Recommender/
├── app.py            # Interactive Dash dashboard
├── requirements.txt
├── README.md
├── report.html        # Exported analysis report
├── notebooks/         # Model development notebooks
└── src/                # Supporting scripts
```

## How to Run
1. Download the MovieLens 1M dataset from https://grouplens.org/datasets/movielens/1m/ and extract it into `data/ml-1m/`
2. Install dependencies: `pip install -r requirements.txt`
3. Explore the analysis: `jupyter notebook` then open `notebooks/01_data_exploration.ipynb`
4. Run the dashboard: `python app.py` then open `http://localhost:8050`

## Key Findings
1. Rating 4 is most common - users tend to rate positively
2. 91.46% matrix sparsity - most users have not rated most movies  
3. Movie Mean baseline performs well (RMSE: 0.976)
4. Content-based filtering successfully identifies genre similarities

## Technologies Used
- Python 3.x
- Pandas, NumPy, Scikit-learn, SciPy
- Matplotlib, Seaborn
- Dash (interactive dashboard)

## Author
Nirmmal Hriday NR - November 2025
