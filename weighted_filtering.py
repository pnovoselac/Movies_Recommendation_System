# Data Manipulation
import pandas as pd
import numpy as np
from math import sqrt

# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Utilities
from ast import literal_eval
from collections import Counter
import datetime
import pickle
from datetime import datetime, timedelta
from itertools import product

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Surprise Library
from surprise import (Reader, Dataset, SVD, SVDpp, KNNBasic, NMF, CoClustering, SlopeOne,
                      NormalPredictor, KNNBaseline, KNNWithMeans, KNNWithZScore, BaselineOnly,
                      accuracy)
from surprise.model_selection import cross_validate
from surprise.model_selection import RandomizedSearchCV
import joblib
from tqdm import tqdm


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval

movies_df = pd.read_csv('new_movies_dataset.csv')
movies_df['bag_of_words'] = movies_df['bag_of_words'].fillna('')

movies_df = movies_df[(movies_df['vote_average'].notnull()) | (movies_df['vote_count'].notnull())]

R = movies_df['vote_average']

v = movies_df['vote_count']

m = movies_df['vote_count'].quantile(0.9)


C = movies_df['vote_average'].mean()


movies_df['weighted_average'] = (R * v + C * m) / (v + m)


current_year = 2020 
movies_df['time_decay_factor'] = 1 / (current_year - movies_df['year'] + 1)

max_revenue = movies_df['revenue'].max()
min_revenue = movies_df['revenue'].min()
movies_df['normalized_revenue'] = (movies_df['revenue'] - min_revenue) / (max_revenue - min_revenue)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(
    movies_df[['popularity', 'weighted_average', 'time_decay_factor', 'normalized_revenue']]
)

weighted_df = pd.DataFrame(
    scaled,
    columns=['popularity', 'weighted_average', 'time_decay_factor', 'normalized_revenue']
)

weighted_df.index = movies_df['id']

weighted_df['score'] = (
    weighted_df['weighted_average'] * 0.4 +
    weighted_df['popularity'] * 0.4 +
    weighted_df['time_decay_factor'] * 0.05 +
    weighted_df['normalized_revenue'] * 0.15
)

weighted_df.to_csv('weighted_df.csv', index=False)

weighted_df_sorted = weighted_df.sort_values(by='score', ascending=False)

top_10_movies = weighted_df_sorted.head(10)
print(top_10_movies)

result_df = pd.merge(top_10_movies, movies_df[['id','original_title', 'year', 'revenue']], on='id', how='left')

recommened_weighted = result_df[['original_title', 'year', 'revenue', 'score']]
print("Weighted Filtering")
print(recommened_weighted)