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
from surprise import (Reader, Dataset, SVD, SVDpp, KNNBasic,
                      NormalPredictor, KNNBaseline, KNNWithMeans, KNNWithZScore, BaselineOnly,
                      accuracy)
from surprise.model_selection import cross_validate
from surprise.model_selection import RandomizedSearchCV
import joblib
from tqdm import tqdm


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


df2 = pd.read_csv('new_movies_dataset.csv')
C= df2['vote_average'].mean()
print(C)

m= df2['vote_count'].quantile(0.9)
print(m)

q_movies = df2.copy().loc[df2['vote_count'] >= m]
print(q_movies.shape)

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)

q_movies['score'] = q_movies.apply(weighted_rating, axis=1)



q_movies = q_movies.sort_values('score', ascending=False)

print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))

pop= df2.sort_values('popularity', ascending=False)
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(6),pop['popularity'].head(6), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")
plt.show()