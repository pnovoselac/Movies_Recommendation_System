#Data Manipulation
import pandas as pd
import numpy as np
from math import sqrt

#Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt

#Utilities
from ast import literal_eval
from collections import Counter
import datetime
import pickle
from datetime import datetime, timedelta

#Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Surprise Library
from surprise import (Reader, Dataset, SVD, SVDpp, KNNBasic, NMF, CoClustering, SlopeOne,
                      NormalPredictor, KNNBaseline, KNNWithMeans, KNNWithZScore, BaselineOnly,
                      accuracy, accuracy)
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

#Učitavanje skupova podataka
movies = pd.read_csv('Dataset/movies.csv')
keywords = pd.read_csv('Dataset/keywords.csv')
credits = pd.read_csv('Dataset/credits.csv')
ratings_df = pd.read_csv('Dataset/ratings_small.csv')

#Spajanje skupova u jedan file
movies_df = movies[~(movies['id'].str.contains('-'))]
movies_df['id'] = movies_df['id'].astype('int64')
movies_df = pd.merge(movies_df, keywords, on='id', how='left')
movies_df = pd.merge(movies_df, credits, on='id', how='left')

#Uklanjanje nepotrebnih značajki
column_to_drop = ['belongs_to_collection', 'homepage', 'status', 'video']
movies_df = movies_df.drop(column_to_drop, axis=1)

#Zamijena nedostajućih vrijednosti 
text_columns = ['original_language','tagline']
movies_df[text_columns] = movies_df[text_columns].fillna('Unknown')

number_columns = ['runtime']
movies_df[number_columns] = movies_df[number_columns].fillna(0)

#Uklanjanje redova sa svim nedostajućim vrijednostima
movies_df.dropna(inplace=True)
movies_df.head()


#Transformacija podataka
movies_df['year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year

movies_df['year'] = movies_df['year'].astype(int)

movies_df['release_date'] = pd.to_datetime(movies_df['release_date'], format="%Y-%m-%d", errors='coerce')

movies_df['popularity'] = movies_df['popularity'].astype('float64')

ratings_df['timestamp'] = ratings_df['timestamp'].apply(lambda x: datetime.fromtimestamp(x))

def extend_list_from_column(df, column_name, target_list, key='name'):
    temp_df = df[column_name].fillna('[]').apply(literal_eval).apply(lambda x: [i[key] for i in x] if isinstance(x, list) else [])
    for i in temp_df:
        if i:
            target_list.extend(i)

def get_director(x):
    return next((i['name'] for i in x if i['job'] == 'Director'), np.nan)

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        return names[:3] if len(names) > 3 else names
    return []

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

def create_bag_of_words(x):
    return ' '.join(x['keywords'] + x['cast'] + [x['director']] + x['genres'])

spoken_languages_list = []
cast_list = []
crew_list = []
company_list = []
country_list = []
original_language_list = []

for i in movies_df['original_language']:
    original_language_list.extend(i.split(', '))

extend_list_from_column(movies_df, 'spoken_languages', spoken_languages_list)
extend_list_from_column(movies_df, 'cast', cast_list, key='character')
extend_list_from_column(movies_df, 'crew', crew_list)
extend_list_from_column(movies_df, 'production_companies', company_list)
extend_list_from_column(movies_df, 'production_countries', country_list)

for col in ['crew', 'cast', 'genres', 'keywords', 'production_companies']:
    movies_df[col] = movies_df[col].fillna('[]').apply(literal_eval)

movies_df['director'] = movies_df['crew'].apply(get_director)

for col in ['cast', 'genres', 'keywords', 'production_companies']:
    movies_df[col] = movies_df[col].apply(get_list)
    
features = ['adult', 'cast', 'keywords', 'director', 'genres', 'production_companies']
for feature in features:
    movies_df[feature] = movies_df[feature].apply(clean_data)
    
movies_df['bag_of_words'] = movies_df.apply(create_bag_of_words, axis=1)

movies_df.to_csv('new_movies_dataset.csv', index=False)
ratings_df.to_csv('new_ratings_dataset.csv', index=False)


#Stupčasti graf izdanih filmova po godini
plt.figure(figsize=(15, 7))
sns.histplot(data=movies_df, x='release_date', bins=30, color='#fdc100', facecolor='#06837f', kde=True)
plt.title('Distribution of Movies by Release Year', fontsize=18)  
plt.xlabel('Release Date', fontsize=14)  
plt.ylabel('Number of Movies', fontsize=14)  
plt.show()


#Prikaz pet najčešćih žanrova 
genres_count = movies_df['genres']
genres_count = genres_count.apply(lambda x: ', '.join(x))
genres_count = Counter(', '.join(genres_count).split(', '))
df_top5 = pd.DataFrame(genres_count.most_common(5), columns=['genre', 'total'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.barplot(data=df_top5, x='genre', y='total', ax=axes[0], palette=['#06837f', '#02cecb', '#b4ffff', '#f8e16c', '#fed811'])
axes[0].set_title('Top 5 Genres in Movies', fontsize=18, weight=600, color='#333d29')

df_all = pd.DataFrame(list(genres_count.items()), columns=['genre', 'total']).sort_values('total', ascending=False)

df_top5.loc[len(df_top5)] = {'genre': 'Others', 'total': df_all[5:].total.sum()}

wedges, texts, autotexts = axes[1].pie(df_top5['total'], labels=df_top5['genre'], autopct='%.2f%%',
                                       textprops={'fontsize': 14}, explode=[0, 0, 0, 0, 0, 0.1],
                                       colors=['#06837f', '#02cecb', '#b4ffff', '#f8e16c', '#fed811', '#fdc100'])
axes[1].set_title('Percentage Ratio of Movie Genres', fontsize=18, weight=600, color='#333d29')

for autotext in autotexts:
    autotext.set_color('#1c2541')
    autotext.set_weight('bold')

sns.despine(left=True, bottom=True)
axes[1].axis('off')

plt.show()

#Najčešći glumci u filmovima
plt.subplots(figsize=(12,10))
list1=[]
for i in movies_df['cast']:
    list1.extend(i)
ax=pd.Series(list1).value_counts()[:15].sort_values(ascending=True).plot.barh(width=0.9,color=sns.color_palette('muted',40))
for i, v in enumerate(pd.Series(list1).value_counts()[:15].sort_values(ascending=True).values): 
    ax.text(.8, i, v,fontsize=10,color='white',weight='bold')
plt.title('Actors with highest appearance')
plt.show()


