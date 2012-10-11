
import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C code
cdef extern double c_spearman_for_python(double* a, double* b, np.int_t* byvar, int n)

@cython.boundscheck(False)
@cython.wraparound(False)
def spearmanr_by(np.ndarray[double, ndim=1, mode="c"] a not None,
             np.ndarray[double, ndim=1, mode="c"] b not None,
             np.ndarray[np.int_t, ndim=1, mode="c"] byvar not None):
    """
    Spearman correlation of a vs b by byvar

    Given a data set of x and y, grouped by the byvar, computes
    the spearman correlation for each group, then returns the average correlation
    across groups.

    byvar must be in sorted order.

    param: a -- a 1-d numpy array of np.float64
    param: b -- a 1-d numpy array of np.float64
    param: byvar -- the by groups, np.int64
    """
    cdef int n
    n = a.shape[0]
    return c_spearman_for_python(&a[0], &b[0], &byvar[0], n)

