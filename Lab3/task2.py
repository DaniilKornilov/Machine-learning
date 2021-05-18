import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def view_scores(res, _name, _type, metric):
    plt.plot(list(res.keys()), list(res.values()))
    plt.title(_name + ' ' + _type)
    plt.xlabel("N clusters")
    plt.ylabel(metric)
    plt.show()


csv_names = ['clustering_1.csv', 'clustering_2.csv', 'clustering_3.csv']

for name in csv_names:
    # load data
    X = read_csv(name, sep='\t').to_numpy()
    scale = StandardScaler()
    scaled_data = scale.fit_transform(X)

    X_principal = pd.DataFrame(scaled_data)
    X_principal.columns = ['P1', 'P2']

    # visualize
    i = 0
    if name == 'clustering_1.csv':
        i = 2
    if name == 'clustering_2.csv':
        i = 3
    if name == 'clustering_3.csv':
        i = 5

    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=AgglomerativeClustering(n_clusters=i).fit_predict(X_principal))
    plt.title(name + ' Hierarchical')
    plt.show()

    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=KMeans(n_clusters=i).fit_predict(X_principal))
    plt.title(name + ' KMeans')
    plt.show()

    # find scores

    # DBSCAN score
    # not scaled data for DBSCAN
    X_principal = pd.DataFrame(X)
    X_principal.columns = ['P1', 'P2']

    plt.scatter(X_principal['P1'], X_principal['P2'],
                c=DBSCAN().fit_predict(X_principal))
    plt.title(name + ' DBSCAN')
    plt.show()

    clustering = DBSCAN().fit(X_principal)
    labels = clustering.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)
    if n_clusters_ > 1:
        print(metrics.silhouette_score(X_principal, DBSCAN().fit_predict(X_principal)))
        print(metrics.davies_bouldin_score(X_principal, DBSCAN().fit_predict(X_principal)))
        print(metrics.calinski_harabasz_score(X_principal, DBSCAN().fit_predict(X_principal)))

    # scaled data for KMeans and hierarchical scores
    X_principal = pd.DataFrame(scaled_data)
    X_principal.columns = ['P1', 'P2']
    # KMeans and hierarchical scores
    silhouette_score_k_means = {}
    davies_bouldin_score_k_means = {}
    calinski_harabasz_score_k_means = {}
    silhouette_score_hierarchical = {}
    davies_bouldin_score_hierarchical = {}
    calinski_harabasz_score_hierarchical = {}
    for i in range(2, 10):
        # KMeans
        silhouette_score_k_means[i] = metrics.silhouette_score(X_principal,
                                                               KMeans(n_clusters=i).fit_predict(X_principal))
        davies_bouldin_score_k_means[i] = metrics.davies_bouldin_score(X_principal,
                                                                       KMeans(n_clusters=i).fit_predict(
                                                                           X_principal))
        calinski_harabasz_score_k_means[i] = metrics.calinski_harabasz_score(X_principal,
                                                                             KMeans(n_clusters=i).fit_predict(
                                                                                 X_principal))
        # Hierarchical
        silhouette_score_hierarchical[i] = metrics.silhouette_score(X_principal,
                                                                    AgglomerativeClustering(n_clusters=i).fit_predict(
                                                                        X_principal))
        davies_bouldin_score_hierarchical[i] = metrics.davies_bouldin_score(X_principal,
                                                                            AgglomerativeClustering(
                                                                                n_clusters=i).fit_predict(
                                                                                X_principal))
        calinski_harabasz_score_hierarchical[i] = metrics.calinski_harabasz_score(X_principal,
                                                                                  AgglomerativeClustering(
                                                                                      n_clusters=i).fit_predict(
                                                                                      X_principal))

    # plots for scores k_means
    view_scores(silhouette_score_k_means, name, 'KMeans', "Silhouette score")
    view_scores(davies_bouldin_score_k_means, name, 'KMeans', "Davies-Bouldin score")
    view_scores(calinski_harabasz_score_k_means, name, 'KMeans', "Calinski_Harabasz score")

    # plots for scores hierarchical
    view_scores(silhouette_score_hierarchical, name, 'Hierarchical', "Silhouette score")
    view_scores(davies_bouldin_score_hierarchical, name, 'Hierarchical', "Davies-Bouldin score")
    view_scores(calinski_harabasz_score_hierarchical, name, 'Hierarchical', "Calinski_Harabasz score")
