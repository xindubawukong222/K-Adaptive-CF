from libmf import mf
import numpy as np
import scipy.sparse


def recommend(A, test_set):
    rr, cc, vv = scipy.sparse.find(A)
    data = []
    for i in range(rr.shape[0]):
        data.append([rr[i], cc[i], vv[i]])
    data = np.array(data)
    row, col, record = test_set
    ind = []
    for i in range(row.shape[0]):
        ind.append([row[i], col[i]])
    ind = np.array(ind)
    engine = mf.MF()
    engine.fit(data, maxiter=100)
    res = engine.predict(ind)
    flag = np.ones(row.shape[0])
    print(res)
    return res, flag


if __name__ == '__main__':
    A = np.random.randn(600, 10000)
    A[A < 0] = 0
    print(A)
    all = []
    for i in range(600):
        for j in range(10000):
            all.append([i, j, A[i][j]])
    np.random.shuffle(all)
    data = np.array(all[:90000])
    print(data)
    ind = []
    for i in range(100):
        ind.append([np.random.randint(0, 600), np.random.randint(0, 10000)])
    ind = np.array(ind)
    engine = mf.MF()
    engine.fit(data)
    res = engine.predict(ind)
    print(res)

