import numpy as np
import scipy.sparse
import time
from utils import *


class New_recommender(object):

    def __init__(self, record):
        self.record = record

    def getMAE(self, a, b, p, q):
        cnt = self.row.shape[0]
        tot = 0
        for i in range(cnt):
            x = self.row[i]
            y = self.col[i]
            pred = p[x].dot(q[y].T)
            tot += abs(self.record[i] - pred)
        return tot / cnt

    def fact(self, A, kk, maxiter):
        row, col, val = scipy.sparse.find(A)
        start = get_time()
        m, n = A.shape
        k = kk
        p = np.random.rand(m, k)
        q = np.random.rand(n, k)
        a = np.random.rand(m)
        b = np.random.rand(n)
        gama = 0.01
        lam = 0.05
        for tt in range(maxiter):
            tot_error = 0
            for i in range(row.shape[0]):
                x = row[i]
                y = col[i]
                pred = p[x].dot(q[y].T)
                e = val[i] - pred
                tot_error += abs(e)
                p[x] += gama * (e * q[y] - lam * p[x])
                q[y] += gama * (e * p[x] - lam * q[y])
            gama *= 0.999
            logger.info('tt = %d  error = %s  MAE = %s  cputime = %s' % (
                        tt, tot_error / row.shape[0], self.getMAE(a, b, p, q),
                        get_time() - start))
        return p, q

    def recommend(self, A, row, col, k, maxiter):
        logger.info('k = %d' % k)
        self.row = row
        self.col = col
        p, q = self.fact(A, k, maxiter)
        cnt = row.shape[0]
        res = np.zeros(cnt)
        flag = np.ones(cnt)
        for i in range(cnt):
            res[i] = p[row[i]].dot(q[col[i]])
        return res, flag
