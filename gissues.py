from issue_grabber import get_issues
from elbow import elbow_method
from KMeans_class import *
from top_features import *

import pickle
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans 
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

num_clusters = 3
max_n_to_test = 15
num_top_feats = 10

print("\n" * 10)
try:
    print("Loading cached data...")
    with open("issues_pickled.pkl", "rb") as f:
        corpus = pickle.load(f)
except Exception as err:
    print(err)
    print("Error: Could not locate cached data. Grabbing issues from github.com.")
    corpus = get_issues()


print("Creating model...")
vectorizer = Tfidf(stop_words = 'english', max_features = 20000)
tf_idf = vectorizer.fit_transform(corpus)
tf_idf_norm = normalize(tf_idf)
tf_idf_array = tf_idf_norm.toarray()

df = pd.DataFrame(tf_idf_array, columns = vectorizer.get_feature_names())
df.head()

if num_clusters == -1:
    print("Testing to see how many clusters to create.")
    Y_sklearn = elbow_method(max_n_to_test, tf_idf_array)
    num_clusters = input("How many clusters should we use? (q to quit)\n")
    try:
        num_clusters = int(num_clusters)
        if (num_clusters <= 1):
            raise ValueError("num_clusters must be positive")
    except:
        exit()
else:
    Y_sklearn = elbow_method(1, tf_idf_array, False)

kmeans = KMeans(n_clusters=num_clusters, max_iter=600, algorithm = 'auto')
fitted = kmeans.fit(Y_sklearn)
predicted_values = kmeans.predict(Y_sklearn)

plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=predicted_values, s=50, cmap='viridis')

#centers = fitted.centroids
#plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.6);
plt.show()

dfs = get_top_features_cluster(tf_idf_array, predicted_values, num_top_feats, vectorizer)
i = 1
for dframe in dfs:
    print("\n"*2)
    print("Category " + str(i))
    print(dframe)
    i += 1


