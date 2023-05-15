import numpy as np
from scipy.spatial.distance import cdist


def PCM(X, t_prev, cluster_count, q):
    t_prev /= np.ones((cluster_count, 1)).dot(np.atleast_2d(t_prev.sum(axis=0)))
    t_prev = np.fmax(t_prev, np.finfo(np.float64).eps)
    t_q = t_prev ** q
    X = X.T
    center = t_q.dot(X) / (np.ones((X.shape[1], 1)).dot(np.atleast_2d(t_q.sum(axis=1))).T)

    dist = cdist(X, center).T
    dist = np.fmax(dist, np.finfo(np.float64).eps)
    
    nta = np.sum(t_q * dist ** 2, axis=1) / np.sum(t_q, axis=1)
    
    j_m = (t_q * dist ** 2).sum() + nta.sum() * ((1 - t_prev) ** q).sum() 
    t = (1 + (((dist **2).T / nta).T **(1 / (q - 1)))) **-1
    t /= np.ones((cluster_count, 1)).dot(np.atleast_2d(t.sum(axis=0)))
    return center, t, j_m


def cmeans(X, cluster_count, q, error = 1e-3, max_iter = 300):
    n = X.shape[1]
    t_init = np.random.rand(cluster_count, n)
    t_init /= np.ones((cluster_count, 1)).dot(np.atleast_2d(t_init.sum(axis=0))).astype(np.float64)
    init = t_init.copy()
    t_init = init
    t = np.fmax(t_init, np.finfo(np.float64).eps)
    j_m = np.zeros(0)

    i = 0
    while i < max_iter - 1:
        t2 = t.copy()
        [center, t, J_jm] = PCM(X, t2, cluster_count, q)
        j_m = np.hstack((j_m, J_jm))
        i += 1
        if np.linalg.norm(t - t2) < error:
            break
    error = np.linalg.norm(t - t2)
    return center, t
