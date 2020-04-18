from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def elbow_method(max_n, ipt_arr, test=True):

    if test:
        number_clusters = range(1, max_n)
        sklearn_pca = PCA(n_components = 2)
        Y_sklearn = sklearn_pca.fit_transform(ipt_arr)

        kmeans = [KMeans(n_clusters=i, max_iter = 600) for i in number_clusters]
        kmeans

        score = [kmeans[i].fit(Y_sklearn).score(Y_sklearn) for i in range(len(kmeans))]
        score

        plt.plot(number_clusters, score)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Score')
        plt.title('Elbow Method')
        plt.show()
    else:
        sklearn_pca = PCA(n_components = 2)
        Y_sklearn = sklearn_pca.fit_transform(ipt_arr)

    return Y_sklearn