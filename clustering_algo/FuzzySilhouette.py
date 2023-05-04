from sklearn.metrics import pairwise_distances
import numpy as np

def FuzzySilhouette(X, U, n_clusters, alpha=2):
    dist_matrix = pairwise_distances(X)
    silhouette = np.zeros(len(X))
    weight = np.zeros(len(X))
    for i in range(len(X)):
        curr_membership = U[:, i]
        if curr_membership[0] > curr_membership[1]:
            max_val = curr_membership[0]
            max_pos = 0
            sec_max = curr_membership[1]
            sec_max_pos = 1
        else:
            max_val = curr_membership[1]
            max_pos = 1
            sec_max = curr_membership[0]
            sec_max_pos = 0
        for j in range(2, n_clusters):
            if curr_membership[j] > max_val:
                sec_max = max_val
                sec_max_pos = max_pos
                max_val = curr_membership[j]
                max_pos = j
            elif sec_max < curr_membership[j] != max_val:
                sec_max = curr_membership[j]
                sec_max_pos = j
        weight[i] = (max_val - sec_max) ** alpha
        a = np.sum(np.multiply(dist_matrix[i], U[max_pos])) / (len(X) - 1)
        b = np.sum(np.multiply(dist_matrix[i], U[sec_max_pos])) / (len(X) - 1)
        silhouette[i] = (b - a) / max(b, a)
        return np.sum(np.multiply(weight, silhouette)) / np.sum(weight)


def find_best_cluster_count(X, max_count, func):
    best_count = 1
    best_score = -1
    all_scores = list()
    for num in range(3, max_count + 1):
        centers, L = func(X.T, 3, 2)
        silh = FuzzySilhouette(X, L, num)
        all_scores.append(silh)
        if silh > best_score:
            best_count = num
            best_score = silh
    return best_count, best_score, all_scores
