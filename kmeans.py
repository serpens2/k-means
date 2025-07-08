import numpy as np
import pandas as pd

def kmeans(X:np.ndarray, K:int, N_ITER:int=100):
    """
    Each element in X represents an observation
    K is the number of clusters
    N_ITER is the number of iterations in Lloyd's algorithm
    """
    clusters = []
    WCSSs = [] # within cluster sum of squares for each clusterization
    for _ in range(N_ITER):
        empty_cluster = False
        # choosing k centroids at random
        rand_indices = np.random.randint(len(X), size=K)
        # ensuring that no two centroids coincide
        while len(set(rand_indices)) != len(rand_indices):
            rand_indices = np.random.randint(len(X), size=K)
        K_means = [X[i] for i in rand_indices]
        # updating clusters until they converge
        for n in range(100):
            cluster_labels = np.array(
                [np.argmin([np.linalg.norm(X_i - K_mean) for K_mean in K_means]) for X_i in X]
            )
            K_means_new = []
            for j in range(K):
                cluster_j = np.array([X_i for X_i, lbl in zip(X, cluster_labels) if lbl == j])
                if len(cluster_j) > 1:
                    K_means_new.append( np.mean(cluster_j,axis=0) )
                else:
                    # if the outlier point was chosen as centroid, it may happen that
                    # the corresponding cluster has only one point. In this case, we
                    # just pass to the next iteration
                    empty_cluster = True
            if empty_cluster:
                break
            # convergence criterion
            M = max([np.linalg.norm(K_means[i] - K_means_new[i]) for i in range(K)])
            K_means = np.copy(K_means_new)
            if M <= 0.001:
                break
        if empty_cluster:
            break
        clusters.append(cluster_labels)
        WCSS = 0
        for j in range(K):
            cluster_j = np.array([X_i for X_i, lbl in zip(X, cluster_labels) if lbl == j])
            WCSS += np.sum([ np.linalg.norm(X_i - K_means[j])**2 for X_i in cluster_j ]) / len(cluster_j)
        WCSSs.append(WCSS)
    # returning clusterization with the lowest WCSS, with the value of WCSS
    best_idx = np.argmin(WCSSs)
    best_clusters = clusters[best_idx]
    best_WCSS = WCSSs[best_idx]
    return best_clusters, best_WCSS

if __name__ == "__main__":
    df = pd.read_csv('Iris.csv')
    X = df.drop(columns=['Species', 'Id']).values
    clusters, WCSS = kmeans(X,K=3)
    print("Best clusterization:")
    print(clusters)
    print(f"With WCSS = {WCSS}")