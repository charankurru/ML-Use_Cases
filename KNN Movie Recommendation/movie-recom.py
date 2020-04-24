# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing datasets
movie_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

#merging datasets
movie_ratings = pd.merge(ratings_df,movie_df,on = 'movieId')
movie_ratings = movie_ratings.drop(["timestamp","genres"],axis = 1)
movie_ratings = movie_ratings.dropna(axis = 0, subset = ['title'])

# finding total number of ratings per each movie
movie_ratingCount = (movie_ratings.
     groupby(by = ['title'])['rating'].
     count().
     reset_index().
     rename(columns = {'rating': 'totalRatingCount'})
     [['title', 'totalRatingCount']]
    )
movie_ratingCount.head()

# merging the total number of ratings per each movie dataframe to movie_ratings dataset
movie_ratings = movie_ratings.merge(movie_ratingCount,on = 'title')

# filtering the data by making an threshold value
movie_ratings = movie_ratings.query('totalRatingCount >= 50')

#converting the dataset into pivot table
movie_ratings_pivot = movie_ratings.pivot_table(index = 'title',columns = 'userId',values = 'rating').fillna(0)

#converting pivot table into sparse matrix
from scipy.sparse import csr_matrix
movie_ratings_matrix = csr_matrix(movie_ratings_pivot.values)
movie_ratings_matrix.head()

#fitting the knn model ie NearestNeighbours
from sklearn.neighbors import NearestNeighbors
model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(movie_ratings_matrix)

# predicting the results
query_index = np.random.choice(movie_ratings_pivot.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie_ratings_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)
# =============================================================================
# print(distances)
# print(distances.flatten())
# print(movie_ratings_pivot.iloc[query_index,:].values.reshape(1,-1))
# 
# =============================================================================

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(movie_ratings_pivot.index[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_ratings_pivot.index[indices.flatten()[i]], distances.flatten()[i]))