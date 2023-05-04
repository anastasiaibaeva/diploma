import numpy as np

def init_memval(cluster_n, data_n):  
    U = np.random.random((cluster_n, data_n))
    val = sum(U)
    U = np.divide(U,np.dot(np.ones((cluster_n,1)), np.reshape(val,(1,data_n))))
    return U

    
def fcm(data, cluster_n, expo = 2, min_err = 0.001, max_iter = 500):
    np.random.seed(0)
    U_old={}
    data_n = data.shape[0]
    U = init_memval(cluster_n, data_n)
    for i in range(max_iter):
        mf = np.power(U,expo)
        center = np.divide(np.dot(mf,data), (np.ones((data.shape[1], 1)) * sum(mf.T)).T)
        diff = np.zeros((center.shape[0], data.shape[0]))
        if center.shape[1] > 1:
            for k in range(center.shape[0]):
                diff[k, :] = np.sqrt(sum(np.power(data-np.dot(np.ones((data.shape[0], 1)), np.reshape(center[k, :], (1, center.shape[1]))),2).T))
        else:
            for k in range(center.shape[0]):
                diff[k, :] = abs(center[k]-data).T
        dist=diff+0.0001;
        num = np.power(dist,(-2/(expo-1)))
        U = np.divide(num, np.dot(np.ones((cluster_n, 1)), np.reshape(sum(num), (1, num.shape[1]))) + 0.0001)
        U_old[i]=U;
        if i> 0:
            if abs(np.amax(U_old[i] - U_old[i-1])) < min_err:
                break
    return center, U