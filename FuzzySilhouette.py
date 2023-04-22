def distance(a, b):
    return np.sum(a != b)

def FuzzySilhouette(X, U, alpha=1):
    n = len(X)
    k = len(U[0])
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            D[i][j] = distance(X[i],X[j])
    obj = np.array([np.argmax(i) for i in U])
    degrees = np.zeros(n)
    a = np.zeros(n)
    b = np.zeros(n)
    sil = np.zeros(n)
    for i in range(2):
        degrees[i] = (np.max(U[i])- np.min(U[i]))**alpha
        B = np.zeros(k)
        i2 = np.argwhere(obj != obj[i]) 
        c2 = np.unique(obj[i2])  # all remaining clusters that object i does not belong
        for c in c2:
            i3 = np.argwhere (obj == c) # objects in the cth cluster
            for object_index in i3:
                B[c] += distance(X[i],X[object_index])
            B[c]/= len(i3)
        B[obj[i]] = float('inf')
        b[i] = np.min(B)
        m = np.argwhere(obj == obj[i]) # # objects in the same cluster of i
        for object_index in m:
            a[i] += distance(X[i],X[object_index])
        a[i]/= len(m)
        if (len(obj[i] == obj)) > 1 and (a[i] >0 or b[i]>0):
            sil[i] = (b[i] - a[i])/max(a[i], b[i])
        else: sil[i] = a[i]
    return sum(degrees * sil)/sum(degrees)
