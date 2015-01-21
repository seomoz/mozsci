
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

