import scipy.sparse.linalg
from k_adaptive_rPCA_recommender import calc
from scipy.sparse import csc_matrix
from utils import *


class old_recommender(object):

    def __init__(self):
        pass

    def recommend(self, AA, test_set, k):
        A = AA.copy()
        B = A.toarray()
        ff = B.sum(axis=1).reshape(-1) / A.getnnz(axis=1)
        for i in range(B.shape[0]):
            x = B[i, :]
            x[x == 0] = ff[i]
        mean = B.mean()
        B -= mean
        B = csc_matrix(B)
        logger.info('start svds.')
        U, S, Vt = scipy.sparse.linalg.svds(B, k)
        logger.info('end svds.')
        mae, result, flag = calc(mean, B, (U, S, Vt), test_set, True)
        result += mean
        return result, flag
