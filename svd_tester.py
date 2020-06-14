from dataset import get_matrix
from k_adaptive_rPCA import k_adaptive_rPCA, randQB_EI
import scipy.sparse.linalg
from utils import *
from logger import *
import numpy as np


class SvdTester(object):

    def __init__(self, dataset_name):
        self.log_file_name = 'log/SvdTester_log-' + 'dataset_name(' + \
                             dataset_name + ')' + '-time(' + \
                             str(time.localtime()) + ').log'
        set_output_file(self.log_file_name)
        logger.info(self.log_file_name + '\n\n')
        self.A = get_matrix(dataset_name)
        self.k = 0

    def test_k_adaptive_rPCA(self):
        logger.info('\n\nTest randSVD q = 10:\n')
        start = get_time()
        u, s, vt = k_adaptive_rPCA(self.A, relerr=0.5, b=20, q=10)
        self.k = u.shape[1]
        logger.info('\n' + 's:')
        logger.info(s.tolist())
        logger.info('\nTester total time: %s\n' % (get_time() - start))

    def test_randQB_EI(self):
        logger.info('\n\nTest randQB_EI:\n')
        start = get_time()
        u, s, vt = randQB_EI(self.A, relerr=0.5, b=20, P=4)
        logger.info('\n' + 's:')
        logger.info(s.tolist())
        logger.info('\nTester total time: %s\n' % (get_time() - start))

    def test_svds(self):
        logger.info('\n\nTest svds(%d):\n' % self.k)
        start = get_time()
        u, s, vt = scipy.sparse.linalg.svds(self.A, self.k)
        logger.info('\n' + 's:')
        order = range(len(s) - 1, -1, -1)
        ss = s[order]
        E = scipy.sparse.linalg.norm(self.A) ** 2
        threshold = 0.5 ** 2 * E
        # print(E, threshold)
        k = 0
        M = u.dot(np.diag(s))
        N = vt
        A = self.A.copy() - M.dot(N)
        for k in range(self.k - 1, -1, -1):
            # tmp = M[:, :k + 1].dot(N[:k + 1, :])
            # print(np.linalg.norm(self.A - tmp) ** 2)
            # if np.linalg.norm(self.A - tmp) ** 2 > threshold:
            #     break
            # print(np.linalg.norm(A) ** 2)
            if np.linalg.norm(A) ** 2 > threshold:
                break
            x = M[:, k].reshape(M.shape[0], 1)
            y = N[k, :].reshape(1, N.shape[1])
            A += x.dot(y)
        k += 2
        logger.info('k = %d\n' % k)
        logger.info(s.tolist())
        logger.info('\nTester total time: %s\n' % (get_time() - start))

    def test(self):
        self.test_k_adaptive_rPCA()
        # self.test_randQB_EI()
        self.test_svds()


if __name__ == '__main__':
    tester = SvdTester('movielens_1M')
    tester.test()
