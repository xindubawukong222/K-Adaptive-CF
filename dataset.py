import numpy as np
from scipy.sparse import csc_matrix
from logger import *


def load_movielens2(small=False):
    path = 'dataset/movielens_1M.csv' if small else 'dataset/movielens_20M.csv'
    f = open(path, 'r')
    lines = f.readlines()
    f.close()
    lines = lines[1:]
    user_index = {}
    user_cnt = 0
    movie_index = {}
    movie_cnt = 0
    row = []
    col = []
    val = []
    for line in lines:
        user, movie, score = line.split(',')[:3]
        if user not in user_index:
            user_index[user] = user_cnt
            user = user_cnt
            user_cnt += 1
        else:
            user = user_index[user]
        if movie not in movie_index:
            movie_index[movie] = movie_cnt
            movie = movie_cnt
            movie_cnt += 1
        else:
            movie = movie_index[movie]
        score = float(score)
        row.append(user)
        col.append(movie)
        val.append(score)
    row = np.array(row)
    col = np.array(col)
    val = np.array(val)
    logger.info('length: %s %s %s' % (row.shape, col.shape, val.shape))
    logger.info('userid range: [%d, %d]' % (row.min(), row.max()))
    logger.info('movieid range: [%d, %d]' % (col.min(), col.max()))
    logger.info('rating range: [%.2f, %.2f]' % (val.min(), val.max()))
    A = csc_matrix((val, (row, col)))
    if A.shape[0] > A.shape[1]:
        A = A.T
    logger.info(str(type(A)) + ' ' + str(A.dtype) + ' ' +
                str(A.shape) + ' ' + str(A.getnnz()))
    return A


def load_movielens_small():
    return load_movielens2(small=True)


def load_book_crossing():
    f = open('dataset/BookCrossing.csv', 'r')
    lines = f.readlines()
    f.close()
    lines = lines[1:]
    user_index = {}
    user_cnt = 0
    movie_index = {}
    movie_cnt = 0
    row = []
    col = []
    val = []
    for line in lines:
        user, movie, score = line.split(';')
        score = float(score[1:-3])
        if score == 0:
            continue
        if user not in user_index:
            user_index[user] = user_cnt
            user = user_cnt
            user_cnt += 1
        else:
            user = user_index[user]
        if movie not in movie_index:
            movie_index[movie] = movie_cnt
            movie = movie_cnt
            movie_cnt += 1
        else:
            movie = movie_index[movie]
        row.append(user)
        col.append(movie)
        val.append(score)
    row = np.array(row)
    col = np.array(col)
    val = np.array(val)
    logger.info('length: %s %s %s' % (row.shape, col.shape, val.shape))
    logger.info('userid range: [%d, %d]' % (row.min(), row.max()))
    logger.info('movieid range: [%d, %d]' % (col.min(), col.max()))
    logger.info('rating range: [%.2f, %.2f]' % (val.min(), val.max()))
    A = csc_matrix((val, (row, col)))
    if A.shape[0] > A.shape[1]:
        A = A.T
    logger.info(str(type(A)) + ' ' + str(A.dtype) + ' ' +
                str(A.shape) + ' ' + str(A.getnnz()))
    A = A[A.getnnz(axis=1) > 2, :]
    A = A[:, A.getnnz(axis=0) > 1]
    A = A[A.getnnz(axis=1) > 0, :]
    logger.info('Deleted some rows and columns.')
    logger.info(str(type(A)) + ' ' + str(A.dtype) + ' ' +
                str(A.shape) + ' ' + str(A.getnnz()))
    return A


def load_hetrec2011():
    f = open('dataset/hetrec2011.dat', 'r')
    lines = f.readlines()
    f.close()
    user_index = {}
    user_cnt = 0
    movie_index = {}
    movie_cnt = 0
    row = []
    col = []
    val = []
    for line in lines[1:]:
        user, movie, score = line.split('\t')[:3]
        if user not in user_index:
            user_index[user] = user_cnt
            user = user_cnt
            user_cnt += 1
        else:
            user = user_index[user]
        if movie not in movie_index:
            movie_index[movie] = movie_cnt
            movie = movie_cnt
            movie_cnt += 1
        else:
            movie = movie_index[movie]
        score = float(score)
        row.append(user)
        col.append(movie)
        val.append(score)
    row = np.array(row)
    col = np.array(col)
    val = np.array(val)
    logger.info('length: %s %s %s' % (row.shape, col.shape, val.shape))
    logger.info('userid range: [%d, %d]' % (row.min(), row.max()))
    logger.info('movieid range: [%d, %d]' % (col.min(), col.max()))
    logger.info('rating range: [%.2f, %.2f]' % (val.min(), val.max()))
    A = csc_matrix((val, (row, col)))
    if A.shape[0] > A.shape[1]:
        A = A.T
    logger.info(str(type(A)) + ' ' + str(A.dtype) + ' ' +
                str(A.shape) + ' ' + str(A.getnnz()))
    return A


def get_matrix_index():
    matrix_index = {
        'movielens_20M': load_movielens2,  # (26744, 138493) 20000263
        'movielens_1M': load_movielens_small,  # (610, 9724) 100836
        'book_crossing': load_book_crossing,  # (105283, 340556) 1149780
                                              # -> (22568, 48631) 249533
        'hetrec2011': load_hetrec2011,  # (2113, 10109) 855598
    }
    return matrix_index


def get_matrix(dataset_name):
    logger.info('Loading matrix %s...' % dataset_name)
    matrix_index = get_matrix_index()
    if dataset_name not in matrix_index:
        logger.info('There\'s no such dataset!')
        return None
    matrix = matrix_index[dataset_name]()
    logger.info('Matrix loaded successfuly!\n')
    return matrix


if __name__ == '__main__':
    index = get_matrix_index()
    for matrix in index:
        get_matrix(matrix)
