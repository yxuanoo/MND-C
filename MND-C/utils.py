import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def set_rows(csr_mat, rows, value=0):
    """
    稀疏矩阵的某些行值置为value
    """
    if not isinstance(csr_mat, csr_matrix):
        csr_mat = csr_mat.tocsr()
    for row in rows:
        csr_mat.data[csr_mat.indptr[row]:csr_mat.indptr[row+1]] = value
    if value == 0:
        csr_mat.eliminate_zeros()
    return csr_mat


def set_columns(csc_mat, cols, value=0):
    """
    稀疏矩阵的某些列值置为value
    """
    if not isinstance(csc_mat, csc_matrix):
        csc_mat = csc_mat.tocsc()
    for col in cols:
        csc_mat.data[csc_mat.indptr[col]:csc_mat.indptr[col+1]] = value
    if value == 0:
        csc_mat.eliminate_zeros()
    return csc_mat


def index_of(array, idx_value):
    """查找array中的每个值在idx_value中对应的索引位置。
    Args:
        array: numpy.array
        idx_value: 一维numpy.array
    """
    index = np.argsort(idx_value)
    sorted_idx_value = idx_value[index]
    sorted_index = np.searchsorted(sorted_idx_value, array)
    array_index = np.take(index, sorted_index, mode="clip")
    mask = idx_value[array_index] != array
    if mask.any():
        return np.ma.array(array_index, mask=mask)
    else:
        return array_index