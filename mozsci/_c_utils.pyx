
cimport cython
cimport numpy as np
np.import_array()

import numpy as np

@cython.boundscheck(False)
@cython.cdivision(True)
def histogram1d_update(
    np.ndarray[np.float64_t, ndim=1] data,
    np.ndarray[np.int64_t, ndim=1] bin_count,
    double bin_width,
    int bins1,
    float mn):
    cdef int ndata = len(data)
    cdef int i
    cdef int bin_index

    for i in range(ndata):
        bin_index = int((data[i] - mn) / bin_width)
        bin_index = min(max(bin_index, 0), bins1)
        bin_count[bin_index] += 1


@cython.boundscheck(False)
@cython.cdivision(True)
def histogram1d_update_counts(
    np.ndarray[np.float64_t, ndim=1] data,
    np.ndarray[np.int64_t, ndim=1] bin_count,
    double bin_width,
    int bins1,
    float mn,
    np.ndarray[np.float64_t, ndim=1] counts):
    cdef int ndata = len(data)
    cdef int i
    cdef int bin_index

    for i in range(ndata):
        bin_index = int((data[i] - mn) / bin_width)
        bin_index = min(max(bin_index, 0), bins1)
        bin_count[bin_index] += <long long>counts[i]


@cython.boundscheck(False)
@cython.cdivision(True)
def histogram1d_compute_indices(
    np.ndarray[np.float64_t, ndim=1] data,
    double bin_width,
    int bins1,
    float mn,
    np.ndarray[np.int64_t, ndim=1] bin_index):
    cdef int ndata = len(data)
    cdef int i
    cdef int this_index

    for i in range(ndata):
        this_index = int((data[i] - mn) / bin_width)
        bin_index[i] = min(max(this_index, 0), bins1)


@cython.boundscheck(False)
@cython.cdivision(True)
def c_auc_wmw(
    np.ndarray[np.int64_t, ndim=1] idxp,
    np.ndarray[np.int64_t, ndim=1] idxn,
    np.ndarray[np.float64_t, ndim=1] parr,
    np.ndarray[np.float64_t, ndim=1] warr):

    cdef int i, j
    cdef double auc = 0.0
    cdef double sum_weights = 0.0
    cdef int nidxp = len(idxp)
    cdef int nidxn = len(idxn)
    cdef double this_weight
    for i in range(nidxp):
        for j in range(nidxn):
            this_weight = warr[idxp[i]] + warr[idxn[j]]
            sum_weights += this_weight
            if parr[idxp[i]] - parr[idxn[j]] > 0.0:
                auc += this_weight

    return auc / sum_weights

