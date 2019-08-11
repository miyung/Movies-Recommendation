import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from surprise import Dataset, SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from surprise.model_selection import cross_validate, KFold
print('Imports successful!')
%matplotlib inline

# Load the MovieLens data.
data = Dataset.load_builtin('ml-100k')
print('Data load successful!')

# 1. Get the ratings file from the data object
# This is just a filename that has all the data stored in it
ratings_file = data.ratings_file

# 2. Load that table using pandas, a commmon python data loading tool
# We set the column names manually here
col_names = ['user_id', 'item_id', 'rating', 'timestamp']
raw_data = pd.read_table(ratings_file, names=col_names)

# 3. Get the rating column
ratings = raw_data.rating

# 4. Generate a bar plot/histogram of that data
ratings.value_counts().sort_index().plot.bar()
print('Histogram generation successful!')

# Exploring ratings
nbratings = len(ratings)
skewness = ratings.skew()
mean = ratings.mean()
median = ratings.median()
minrating = ratings.min()
maxrating = ratings.max()
var = ratings.var()
std = sqrt(var)
q1 = ratings.quantile(0.25)
q3 = ratings.quantile(0.75)
nbmovies = len(np.unique(raw_data.item_id))
nbusers = len(np.unique(raw_data.user_id))
ratings_per_movie = raw_data.groupby('item_id')['rating'].count()
ratings_per_user = raw_data.groupby('user_id')['rating'].count()


print ('There are ' + str(nbratings) + ' ratings in the dataset.')
print ('There are ' + str(nbmovies) + ' movies in the dataset.')
print ('There are ' + str(nbusers) + ' users in the dataset.')

if skewness > 0:
    print ('The distribution of ratings is positively skewed.')
elif skewness < 0:
    print ('The distribution of ratings is negatively skewed.')
else:
    print ('The distribution of ratings is symmetric, i.e. similar to the normal distribution.')
    
print ('The distribution of ratings has a mean of ' + str(mean) + ', and a median of ' + str(median) + '.')
print ('The distribution of ratings has a minimum value of ' + str(minrating) + ', and a maximum value of ' + str(maxrating) + '.')
print ('The range of the distribution is ' + str(maxrating - minrating) + '.')
print ('The distribution of ratings has a variance of ' + str(var) + ', and a standard deviation of ' + str(std) + '.')
print ('The distribution of ratings has an Interquartile range (IQR) of ' + str(q3 - q1) + '.')

# Generate plots for ratings data
f1 = plt.figure(1)
plt.boxplot(ratings, 0, 'rs', 0)
plt.title('Box plot of ratings')

f2 = plt.figure(2)
plt.plot(ratings_per_movie, 'o', color = 'blue')
plt.ylabel('Number of ratings per movie')
plt.xlabel('Item ID')

f3 = plt.figure(3)
plt.plot(ratings_per_user, 'o', color = 'red')
plt.ylabel('Number of ratings per user')
plt.xlabel('User ID')

plt.show()

# Model 1: Random
# Create model object
model_random = NormalPredictor()
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
model_random_results = cross_validate(model_random, data, measures=['RMSE'], cv=5, verbose=True)
print('Model training successful!')

# Model 2: User-Based Collaborative Filtering
# Create model object
model_user = KNNBasic(sim_options={'user_based': True})
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may have a lot of print output
# You can set verbose=False to prevent this from happening
model_user_results = cross_validate(model_user, data, measures=['RMSE'], cv=5, verbose=True)
print('Model training successful!')

# Model 3: Item-Based Collaborative Filtering
# Create model object
model_item = KNNBasic(sim_options={'user_based': False})
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may have a lot of print output
# You can set verbose=False to prevent this from happening
model_item_results = cross_validate(model_item, data, measures=['RMSE'], cv=5, verbose=True)
print('Model training successful!')

# Model 4: Matrix Factorization
# Create model object
model_matrix = SVD()
print('Model creation successful!')

# Train on data using cross-validation with k=5 folds, measuring the RMSE
# Note, this may take some time (2-3 minutes) to train, so please be patient
model_matrix_results = cross_validate(model_matrix, data, measures=['RMSE'], cv=5, verbose=True)
print('Model training successful!')

# Precision and Recall @ k
# Create a function that takes in some predictions, a value of k and a threshold parameter
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    '''Return precision and recall at k metrics for each user.'''

    # First map the predictions to each user.
    user_est_true = dict()
    for uid, _, true_r, est, _ in predictions:
        current = user_est_true.get(uid, list())
        current.append((est, true_r))
        user_est_true[uid] = current

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls

print('Function creation successful!')

# Compute the precision, recall and f1-score at k = 5 and 10 for each of our 4 models. 
# Use 5-fold cross validation to average the results across the entire dataset.
# Make list of k values
K = [5, 10]

# Make list of models
models = [model_random, model_user, model_item, model_matrix]

# Create k-fold cross validation object
kf = KFold(n_splits=5)

for k in K:
    for model in models:
        print(f'>>> k={k}, model={model.__class__.__name__}')
        # Run folder and take average
        p = []
        r = []
        for trainset, testset in kf.split(data):
            model.fit(trainset)
            predictions = model.test(testset, verbose=False)
            precisions, recalls = precision_recall_at_k(predictions, k=k, threshold=3.5)

            # Precision and recall can then be averaged over all users
            p.append(sum(prec for prec in precisions.values()) / len(precisions))
            r.append(sum(rec for rec in recalls.values()) / len(recalls))
        
        print('>>> precision:', round(sum(p) / len(p), 3))
        print('>>> recall  :', round(sum(r) / len(r), 3))
        print('>>> f1 score:', round(2 * ((sum(p) / len(p)) * (sum(r) / len(r))) / ((sum(p) / len(p)) + (sum(r) / len(r))), 3))
        print('\n')

print('Precision and recall computation successful!')

# Top-n Predictions
# Create a function to get top-n. This project uses n = 5
def get_top_n(predictions, n=5):
    '''Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_n = dict()
    for uid, iid, true_r, est, _ in predictions:
        current = top_n.get(uid, [])
        current.append((iid, est))
        top_n[uid] = current

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

print('Function creation successful!')

# Create training set and test set
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
print('Trainset and testset creation successful!')

# Get top-n predictions for each model
for model in models:
    model.fit(trainset)
    predictions = model.test(testset)
    top_n = get_top_n(predictions, n=5)
    # Print the first one
    user = list(top_n.keys())[1]
    print(f'model: {model}, {user}: {top_n[user]}')

print('Top N computation successful!')