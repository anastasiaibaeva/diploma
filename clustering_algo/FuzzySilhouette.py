from sklearn.metrics import pairwise_distances
import numpy as np

def FuzzySilhouette(X, U, n_clusters, alpha=2):
    distance_matrix = pairwise_distances(X)
    silhouette = np.zeros(len(X))
    weight = np.zeros(len(X))
    for i in range(len(X)):
        curr = U[:, i]
        if curr[0] > curr[1]:
            maximum = curr[0]
            maximum_pos = 0
            second_maximum = curr[1]
            second_maximum_pos = 1
        else:
            maximum = curr[1]
            maximum_pos = 1
            second_maximum = curr[0]
            second_maximum_pos = 0
        for j in range(2, n_clusters):
            if curr[j] > maximum:
                second_maximum = maximum
                second_maximum_pos = maximum_pos
                maximum = curr[j]
                maximum_pos = j
            elif second_maximum < curr[j] != maximum:
                second_maximum = curr[j]
                second_maximum_pos = j
        weight[i] = (maximum - second_maximum) ** alpha
        first = np.sum(np.multiply(distance_matrix[i], U[maximum_pos])) / (len(X) - 1)
        second = np.sum(np.multiply(distance_matrix[i], U[second_maximum_pos])) / (len(X) - 1)
        silhouette[i] = (second - first) / max(second, first)
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
