from utils import *
import numpy as np
from ctypes import *
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


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


#
# relation: m * m
# row: cnt
# col: cnt
# tmp: m
#
def call_cpp(m, cnt, relation, row, col, tmp, A_len, A_row, A_col, A_val, reset):
    logger.info('calling C.')
    foo = CDLL('./calc.so')

    get_result = foo.get_result_
    get_result.argtypes = [c_int, c_int, POINTER(c_double), POINTER(c_int),
                           POINTER(c_int), POINTER(c_double), c_int,
                           POINTER(c_int), POINTER(c_int), POINTER(c_double),
                           c_bool]
    get_result.restype = POINTER(c_double)

    relation2 = np.reshape(relation, -1).astype(np.float64)
    if not relation2.flags['C_CONTIGUOUS']:
        relation2 = np.ascontiguousarray(relation2, dtype=relation2.dtype)
    relation_ptr = relation2.ctypes.data_as(POINTER(c_double))
    row = row.astype(np.int32)
    if not row.flags['C_CONTIGUOUS']:
        row = np.ascontiguousarray(row, dtype=row.dtype)
    row_ptr = row.ctypes.data_as(POINTER(c_int))
    col = col.astype(np.int32)
    if not col.flags['C_CONTIGUOUS']:
        col = np.ascontiguousarray(col, dtype=col.dtype)
    col_ptr = col.ctypes.data_as(POINTER(c_int))
    tmp = tmp.astype(np.float64)
    if not tmp.flags['C_CONTIGUOUS']:
        tmp = np.ascontiguousarray(tmp, dtype=tmp.dtype)
    tmp_ptr = tmp.ctypes.data_as(POINTER(c_double))
    A_row = A_row.astype(np.int32)
    if not A_row.flags['C_CONTIGUOUS']:
        A_row = np.ascontiguousarray(A_row, dtype=A_row.dtype)
    A_row_ptr = A_row.ctypes.data_as(POINTER(c_int))
    A_col = A_col.astype(np.int32)
    if not A_col.flags['C_CONTIGUOUS']:
        A_col = np.ascontiguousarray(A_col, dtype=A_col.dtype)
    A_col_ptr = A_col.ctypes.data_as(POINTER(c_int))
    A_val = A_val.astype(np.float64)
    if not A_val.flags['C_CONTIGUOUS']:
        A_val = np.ascontiguousarray(A_val, dtype=A_val.dtype)
    A_val_ptr = A_val.ctypes.data_as(POINTER(c_double))

    result = get_result(m, cnt, relation_ptr, row_ptr, col_ptr, tmp_ptr,
                        A_len, A_row_ptr, A_col_ptr, A_val_ptr, reset)
    py_result = np.ctypeslib.as_array(result, (cnt,)).astype(np.float64)
    return py_result


def calc(mean, A, SVD, valid_set, reset):
    U, S, Vt = SVD
    transformed = U.dot(np.diag(np.sqrt(S)))
    tmp = np.linalg.norm(transformed, axis=1)
    tmp[tmp == 0] = 1e18
    go = transformed / tmp[:, None]
    relation = go.dot(go.T)
    logger.info('relation: ' + str(relation.shape))
    tmp = scipy.sparse.linalg.norm(A, axis=1)
    row, col, record = valid_set
    cnt = row.shape[0]
    result = np.zeros(cnt)
    flag = np.zeros(cnt)
    A_row, A_col, A_val = scipy.sparse.find(A)
    result = call_cpp(A.shape[0], cnt, relation, row, col, tmp, A_row.shape[0],
                      A_row, A_col, A_val, reset)
    flag[result > 0] = 1
    num = flag.sum()
    tot = (np.abs(record - (result + mean)) * flag).sum()
    # logger.info(str(num) + '    ' + str(tot))
    mae = tot / num
    return mae, result, flag


# A: scipy.sparse.csc_matrix, m <= n
# output: U(k * k), S(1 * k), V(n * k)
# A = U * diag(S) * V'
def k_adaptive_rPCA_recommend(valid_set, test_set, mean, A, b=20, q=11):
    calc_mae = calc
    if b > A.shape[0] / 20:
        b = int(A.shape[0] / 20)
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
    maxiter = min(int(min(m, n) / b), 100)
    min_mae = 1e18
    all_mae = []
    k = 0
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
        U1, S1, Vt1 = eigSVD(B.T)
        U = Q.dot(Vt1.T)
        S = S1
        Vt = U1.T
        reset = True if i == 0 else False
        mae, _, _ = calc_mae(mean, A, (U, S, Vt), valid_set, reset)
        mae2, _, _ = calc_mae(mean, A, (U, S, Vt), test_set, False)
        all_mae.append(((i + 1) * b, mae, mae2))
        logger.info('mae_valid = %s' % mae)
        logger.info('mae_test = %s' % mae2)
        logger.info('process_time: ' + str(get_time() - start2))
        if len(all_mae) > 5:
            tmp_list = [x[1] for x in all_mae[-5:]]
            if min(tmp_list) > min_mae:
                break
        if mae < min_mae:
            min_mae = mae
            k = b * (i + 1)
    logger.info('\nNow running test_set.')
    Q = Q[:, :k]
    B = B[:k, :]
    U1, S1, Vt1 = eigSVD(B.T)
    U = Q.dot(Vt1.T)
    S = S1
    Vt = U1.T
    mae, result, flag = calc_mae(mean, A, (U, S, Vt), test_set, False)
    logger.info('\nall_mae:')
    for xx in all_mae:
        logger.info('size = %d   mae = %s  mae2 = %s' % (xx[0], xx[1], xx[2]))
    logger.info('k = %d' % k)
    return result, flag, k

