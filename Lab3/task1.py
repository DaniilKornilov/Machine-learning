from sklearn.cluster import KMeans
from pandas import read_csv
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

for j in range(0, 2):
    # read, standardize
    X = read_csv('pluton.csv', sep=',').to_numpy()
    scale = StandardScaler()
    scaled_data = scale.fit_transform(X)
    pca = PCA(n_components=2)
    title = 'Non-standardize'
    X_principal = pca.fit_transform(X)
    if j == 1:
        title = 'Standardize'
        X_principal = pca.fit_transform(scaled_data)

    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']

    # visualize
    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=KMeans(n_clusters=3).fit_predict(X_principal))
    plt.title(title)
    plt.show()

    # scores
    silhouette_score = {}
    davies_bouldin_score = {}
    calinski_harabasz_score = {}
    for i in range(1, 10):
        silhouette_score[i] = metrics.silhouette_score(X_principal,
                                                       KMeans(n_clusters=3, max_iter=i).fit_predict(X_principal))
        davies_bouldin_score[i] = metrics.davies_bouldin_score(X_principal,
                                                               KMeans(n_clusters=3, max_iter=i).fit_predict(
                                                                   X_principal))
        calinski_harabasz_score[i] = metrics.calinski_harabasz_score(X_principal,
                                                                     KMeans(n_clusters=3, max_iter=i).fit_predict(
                                                                         X_principal))

    # plots for scores
    plt.plot(list(silhouette_score.keys()), list(silhouette_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Silhouette score")
    plt.show()

    plt.plot(list(davies_bouldin_score.keys()), list(davies_bouldin_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Davies-Bouldin score")
    plt.show()

    plt.plot(list(calinski_harabasz_score.keys()), list(calinski_harabasz_score.values()))
    plt.title(title)
    plt.xlabel("Max iterations")
    plt.ylabel("Calinski_Harabasz score")
    plt.show()
