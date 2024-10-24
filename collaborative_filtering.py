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



ratings_df = pd.read_csv('new_ratings_dataset.csv')
print("Starting to split the data")
train_df, test_df = train_test_split(ratings_df, test_size=0.25, random_state=42)

print("Initializing reader")
reader = Reader(rating_scale=(1, 5))

print("Splitting the data")

train_data = Dataset.load_from_df(train_df[['userId', 'movieId', 'rating']], reader)
test_data = Dataset.load_from_df(test_df[['userId', 'movieId', 'rating']], reader)

print("Building trainset and testset")
trainset = train_data.build_full_trainset()
testset = test_data.build_full_trainset().build_testset()

algorithms = [SVD, KNNWithZScore, SVDpp]


print("Setting parameters")
param_grid = {
    SVD: {
        'n_factors': [125, 150],
        'n_epochs': [25, 30],
        'lr_all': [0.01],
        'reg_all': [0.2]
    },
    KNNWithZScore: {
        'k': [35, 40],
        'min_k': [1, 3],
        'sim_options': {
            'name': ['cosine'],
            'user_based': [True]
        }
    },
    SVDpp: {
        'n_factors': [50, 100],
        'n_epochs': [20],
        'lr_all': [0.005],
        'reg_all': [0.2]
    }
}

best_models = {}
best_params = {}
best_rmse_scores = {}

print("Starting parallel training")
joblib.parallel_backend('loky')


print("Starting RandomizedSearchCV")
for algo in tqdm(algorithms, desc="Algorithms"):
    print(f"Starting grid search for {algo.__name__}")
    
    if algo == SVD:
        n_iter_algo = 4 
    elif algo == KNNWithZScore:
        n_iter_algo = 4 
    elif algo == SVDpp:
        n_iter_algo = 2  
    
    rs = RandomizedSearchCV(algo, param_grid[algo], measures=['rmse'], cv=3, n_iter=n_iter_algo, n_jobs=-1, random_state=42)
    rs.fit(train_data)
    
    best_models[algo.__name__] = rs.best_estimator['rmse']
    best_params[algo.__name__] = rs.best_params['rmse']
    best_rmse_scores[algo.__name__] = rs.best_score['rmse']


print("Done RandomizedSearchCV")
plt.figure(figsize=(10, 6))
plt.bar(best_rmse_scores.keys(), best_rmse_scores.values(), color='blue')
plt.xlabel('Algorithms')
plt.ylabel('Best RMSE Score')
plt.title('Best RMSE Scores by Algorithm')
plt.show()

for algo_name, model in best_models.items():
    print(f"Best Model for {algo_name}:\n", model)
    print(f"Best Parameters for {algo_name}:\n", best_params[algo_name])
    print(f"Best RMSE score for {algo_name}:\n", best_rmse_scores[algo_name])

best_algo = min(best_rmse_scores, key=best_rmse_scores.get)

print(f"Best Algorithm: {best_algo}")
print(f"Best Parameters: {best_params[best_algo]}")
print(f"Best RMSE score: {best_rmse_scores[best_algo]}")

best_algo_name = min(best_rmse_scores, key=best_rmse_scores.get)
best_algo_model = best_models[best_algo_name]

with open('best_model.pkl', 'wb') as f:
    pickle.dump((best_algo_model), f)

print("Best model saved to 'best_model.pkl'")

# Učitavanje najboljeg modela i trainseta iz pickle-a
with open('best_model.pkl', 'rb') as f:
    best_algo_model, trainset = pickle.load(f)

# Treniranje najboljeg algoritma na cijelom trening skupu
best_algo_model.fit(trainset)

# Spremanje najboljeg modela
with open('best_model_fit.pkl', 'wb') as f:
    pickle.dump((best_algo_model, trainset), f)

print("Best trained model and trainset saved to 'best_model_train.pkl'")

best_algo_model.trainset = trainset

predictions = best_algo_model.test(testset)

rmse_score = accuracy.rmse(predictions)

mae_score = accuracy.mae(predictions)
fcp_score = accuracy.fcp(predictions)

print(f"Evaluation Metrics:")
print(f"RMSE score: {rmse_score}")
print(f"MAE score: {mae_score}")
print(f"FCP score: {fcp_score}")

# Save the predictions to a DataFrame
predictions_df = pd.DataFrame(predictions)
predictions_df = predictions_df.rename(columns={'uid': 'userId', 'iid': 'movieId', 'r_ui': 'actual', 'est': 'predicted'})
predictions_df['error'] = np.abs(predictions_df['actual'] - predictions_df['predicted'])

# Save DataFrame to CSV file
predictions_df.to_csv('predictions.csv', index=False)

# Calculate RMSE for the predictions
rmse_score = accuracy.rmse(predictions)

# Calculate other accuracy measures
mae_score = accuracy.mae(predictions)
fcp_score = accuracy.fcp(predictions)

print(f"Evaluation Metrics:")
print(f"RMSE score: {rmse_score}")
print(f"MAE score: {mae_score}")
print(f"FCP score: {fcp_score}")

