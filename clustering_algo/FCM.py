import numpy as np

def init(cluster_count, X):  
    U = np.random.random((cluster_count, X))
    val = sum(U)
    sec = np.dot(np.ones((cluster_count, 1)), np.reshape(val,(1, X)))
    U = np.divide(U, sec)
    return U

    
def fcm(X, cluster_count, exp = 2, min_error = 0.001, max_iter = 500):
    U_prev = {}
    U = init(cluster_count, X.shape[0])
    for i in range(max_iter):
        mf = np.power(U, exp)
        cntr = np.divide(np.dot(mf, X), (np.ones((X.shape[1], 1)) * sum(mf.T)).T)
        diff = np.zeros((cntr.shape[0], X.shape[0]))
        if cntr.shape[1] > 1:
            for k in range(cntr.shape[0]):
                diff[k, :] = np.sqrt(sum(np.power(X - np.dot(np.ones((X.shape[0], 1)), np.reshape(cntr[k, :], (1, cntr.shape[1]))),2).T))
        else:
            for k in range(cntr.shape[0]):
                diff[k, :] = abs(cntr[k] - X).T
        curr = np.power(diff + 0.0001, (-2 / (exp - 1)))
        U = np.divide(curr, np.dot(np.ones((cluster_count, 1)), np.reshape(sum(curr), (1, curr.shape[1]))) + 0.0001)
        U_prev[i] = U
        if i > 0:
            if abs(np.amax(U_prev[i] - U_prev[i-1])) < min_error:
                break
    return cntr, U
