from dataset import get_matrix
from utils import *
import numpy as np
import os
import scipy.sparse.linalg
from k_adaptive_rPCA_recommender import k_adaptive_rPCA_recommend
from RMF_model_2009 import New_recommender
from SVD_model_2000 import old_recommender
import libmf_model


class recommendTester(object):

    def __init__(self, dataset_name):
        self.log_file_name = 'log/recommendTester_log-' + 'dataset_name(' + \
                             dataset_name + ')' + '-time(' + \
                             str(time.localtime()) + ').log'
        set_output_file(self.log_file_name)
        logger.info(self.log_file_name + '\n\n')
        self.A = get_matrix(dataset_name)
        row, col = self.A.nonzero()
        choose = np.arange(row.shape[0])
        np.random.shuffle(choose)
        num = int(len(choose) * 0.05)
        valid = choose[:num]
        test = choose[num:2 * num]
        row1 = row[valid]
        col1 = col[valid]
        record1 = np.asarray(self.A[row1, col1])[0]
        row2 = row[test]
        col2 = col[test]
        record2 = np.asarray(self.A[row2, col2])[0]
        self.A[row1, col1] = 0
        self.A[row2, col2] = 0
        logger.info('\n' + 'valid: %s %s %s'
                    % (row1.shape, col1.shape, record1.shape))
        logger.info('test: %s %s %s' % (row2.shape, col2.shape, record2.shape))
        logger.info('Now there\'re %d nonzero numbers in A.'
                    % (self.A != 0).sum())
        self.valid_set = (row1, col1, record1)
        self.test_set = (row2, col2, record2)
        self.k = 0
        rr, cc, vv = scipy.sparse.find(self.A)
        np.savetxt('input_for_SVT/A_row.txt', rr)
        np.savetxt('input_for_SVT/A_col.txt', cc)
        np.savetxt('input_for_SVT/A_val.txt', vv)
        np.savetxt('input_for_SVT/test_row.txt', row2)
        np.savetxt('input_for_SVT/test_col.txt', col2)
        np.savetxt('input_for_SVT/test_val.txt', record2)

    def test_k_adaptive_rPCA(self):
        start = get_time()
        AA = self.A.copy()
        mean = scipy.sparse.find(self.A)[2].mean()
        AA[AA != 0] -= mean
        result, flag, self.k = k_adaptive_rPCA_recommend(self.valid_set,
                                                           self.test_set, mean,
                                                           AA, b=20, q=11)
        num = flag.sum()
        tot = (np.abs(self.test_set[2] - (result + mean)) * flag).sum()
        mae = tot / num
        logger.info('mae: %s' % mae)
        logger.info('time: %s s' % (get_time() - start))

    def test_old_recommender(self):
        logger.info('\n\nTesting old recommender:\n')
        start = get_time()
        recommender = old_recommender()
        res, flag = recommender.recommend(self.A, self.test_set, self.k)
        num = flag.sum()
        tot = (np.abs(self.test_set[2] - res) * flag).sum()
        mae = tot / num
        logger.info('mae: %s' % mae)
        logger.info('time: %s s' % (get_time() - start))

    def test_new_recommender(self):
        logger.info('\n\nTesting new recommender:\n')
        start = get_time()
        recommender = New_recommender(self.test_set[2])
        res, flag = recommender.recommend(self.A, self.test_set[0],
                                          self.test_set[1], self.k, maxiter=100)
        num = flag.sum()
        tot = (np.abs(self.test_set[2] - res) * flag).sum()
        mae = tot / num
        logger.info('mae: %s' % mae)
        logger.info('time: %s s' % (get_time() - start))

    def test_libmf_recommender(self):
        logger.info('\n\nTesting libmf recommender:\n')
        start = get_time()
        res, flag = libmf_model.recommend(self.A, self.test_set)
        num = flag.sum()
        tot = (np.abs(self.test_set[2] - res) * flag).sum()
        mae = tot / num
        logger.info('mae: %s' % mae)
        logger.info('time: %s s' % (get_time() - start))

    def test(self):
        # self.test_k_adaptive_rPCA()
        # self.test_old_recommender()
        # self.test_new_recommender()
        self.test_libmf_recommender()


if __name__ == '__main__':
    np.random.seed(0)

    tester = recommendTester('book_crossing')
    tester.test()
