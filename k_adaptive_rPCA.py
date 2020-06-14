from utils import *
import scipy
import math
import time
from scipy.sparse import csc_matrix, linalg
import numpy as np


# A: numpy.ndarray, m >= n
# output: U, S, V
def eigSVD(A):
    n = A.shape[1]
    B = A.T.dot(A)
    D, V = np.linalg.eig(B)
    S = np.sqrt(D)
    S2 = np.diag(1 / S)
    U = A.dot(V).dot(S2)
    return U, S, V.T


# A: scipy.sparse.csc_matrix, m <= n
# output: U(k * k), S(1 * k), V(n * k)
# A = U * diag(S) * V'
def k_adaptive_rPCA(A, relerr=0.5, b=20, q=11):
    m, n = A.shape
    if m > n:
        logger.info('randSVD error: randSVD needs m <= n!')
        return
    if q < 2:
        logger.info('randSVD_error: q must >= 2!')
        return
    P = int((q - 1) / 2)
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))
    E = scipy.sparse.linalg.norm(A) ** 2
    threshold = relerr ** 2 * E
    logger.info('\nrelerr = %.5f' % relerr)
    logger.info('E0 = %.5f\nthreshold = %.5f' % (E, threshold))
    maxiter = int(math.ceil(min(m, n) / 2.0 / b))
    for i in range(maxiter):
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~ i = %d ~~~~~~~~~~~~~~~~~~~~~' % i)
        start2 = get_time()
        if q % 2 == 0:
            Omg = np.random.randn(n, b)
            Y = A.dot(Omg) - Q.dot(B.dot(Omg))
            Qi = np.linalg.qr(Y)[0]
        else:
            Qi = np.random.randn(m, b)
        for j in range(P):
            if j == P - 1:
                R = A.T.dot(Qi)
                Qi = np.linalg.qr(A.dot(R) - Q.dot(B.dot(R)))[0]
            else:
                Qi = scipy.linalg.lu(A.dot(A.T.dot(Qi)), permute_l=True)[0]
        Qi = np.linalg.qr(Qi - Q.dot(Q.T.dot(Qi)))[0]
        Bi = (A.T.dot(Qi)).T  # Qi.T.dot(A)
        Q = np.concatenate((Q, Qi), axis=1)
        B = np.concatenate((B, Bi), axis=0)
        E -= np.linalg.norm(Bi) ** 2
        if E < threshold:
            break
        logger.info('E: %s' % E)
        logger.info('process_time: ' + str(get_time() - start2))
    if not E < threshold:
        logger.info('randSVD_sparse wrong!')
        return Q, B, Q.shape[1]
    E = scipy.sparse.linalg.norm(A) ** 2
    k = 0
    for k in range(B.shape[0]):
        E -= np.linalg.norm(B[k]) ** 2
        if E < threshold:
            break
    k += 1
    logger.info('k = %d' % k)
    U1, S1, Vt1 = eigSVD(B.T)
    U = Q.dot(Vt1.T)
    S = S1
    Vt = U1.T
    return U[:, :k], S[:k], Vt[:k, :]


def randQB_EI(A, relerr=0.5, b=20, P=5):
    m, n = A.shape
    Q = np.zeros((m, 0))
    B = np.zeros((0, n))
    k = 0
    E = scipy.sparse.linalg.norm(A)**2
    threshold = relerr**2 * E
    logger.info('E0 = %.5f\nthreshold = %.5f' % (E, threshold))
    maxiter = int(math.ceil(min(m, n) / 2.0 / b))
    flag = False
    tt = True
    for i in range(1, maxiter + 1):
        logger.info('~~~~~~~~~~~~~~~~~~~~~~~~ i = %d ~~~~~~~~~~~~~~~~~~~~~' % i)
        start2 = get_time()
        Omg = np.random.randn(n, b)
        # b = b * 2
        Y = A.dot(Omg) - Q.dot(B.dot(Omg))
        Qi = np.linalg.qr(Y)[0]
        for j in range(1, P + 1):
            Qi = np.linalg.qr(A.T.dot(Qi) - B.T.dot(Q.T.dot(Qi)))[0]
            Qi = np.linalg.qr(A.dot(Qi) - Q.dot(B.dot(Qi)))[0]
        Qi = np.linalg.qr(Qi - Q.dot(Q.T.dot(Qi)))[0]
        Bi = (A.T.dot(Qi)).T - Qi.T.dot(Q).dot(B)
        Q = np.concatenate((Q, Qi), axis=1)
        B = np.concatenate((B, Bi), axis=0)
        temp = E - np.linalg.norm(Bi)**2
        if temp < threshold:
            for j in range(1, b + 1):
                E = E - np.linalg.norm(Bi[j-1, :])**2
                if E < threshold:
                    flag = True
                    k = (i - 1) * b + j
                    break
        else:
            E = temp
            logger.info('E = %.5f' % E)
        if flag:
            break
        logger.info('process_time: ' + str(get_time() - start2))
    if not flag:
        logger.info('randQB_EI wrong!')
        k = Q.shape[1]
    logger.info('k = %d' % k)
    U1, S1, Vt1 = eigSVD(B.T)
    U = Q.dot(Vt1.T)
    S = S1
    Vt = U1.T
    return U[:, :k], S[:k], Vt[:k, :]


def get_sparse_matrix(m, n):
#     row = np.array([0, 2, 2, 0, 1, 2, 0, 0, 0])
#     col = np.array([0, 0, 1, 2, 2, 2, 3, 3, 4])
#     data = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11]).astype(np.float64)
#     return csc_matrix((data, (row, col)))
    return load_movielens_small()


if __name__ == '__main__':
    # a = np.array([[1, 0, 4], [0, 0, 5], [2, 3, 6]])
    # a = np.random.randn(4, 6)
    a = get_sparse_matrix(3, 4)
    logger.info('a:\n', a.toarray())

    start1 = time.time()
    start2 = time.process_time()

    u, s, vt = randQB_EI(a)
    logger.info('u:')
    logger.info(u)
    logger.info('s:')
    logger.info(s)
    logger.info('vt:')
    logger.info(vt)
    logger.info('u*s*vt:')
    b = u.dot(np.diag(s)).dot(vt)
    logger.info('error:', np.linalg.norm(a - b) ** 2)

    # a = a.toarray().T
    # logger.info(a)
    # u, s, vt = eigSVD(a)
    # logger.info('u:')
    # logger.info(u)
    # logger.info('s:')
    # logger.info(s)
    # logger.info('vt:')
    # logger.info(vt)
    # logger.info('u*s*vt:')
    # logger.info(u.dot(np.diag(s)).dot(vt))

    # u, s, vt = scipy.sparse.linalg.svds(a, 117)
    # logger.info('u:')
    # logger.info(u)
    # logger.info('s:')
    # logger.info(s)
    # logger.info('vt:')
    # logger.info(vt)
    # b = u.dot(np.diag(s)).dot(vt)
    # logger.info('error:', np.linalg.norm(a - b) ** 2)

    logger.info('\n\ntime:', time.time() - start1)
    logger.info('process_time:', time.process_time() - start2)
