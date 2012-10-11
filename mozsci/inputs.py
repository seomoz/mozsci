"""Input feature manipulation, including normalizations"""

import numpy as np

def mean_std_weighted(x, weights=None):
    """Computes weighted mean and standard deviation.

    x = a (N, ) or an (N, nx) numpy array
    weights = a (N, ) numpy array of weights or None (no weights)

    Returns {'mean':[means], 'std':[standard deviations]}
    where each value is a len(nx) array for each feature
    """
    if weights is None:
        ret = {'mean': np.mean(x, axis=0), 'std': np.std(x, axis=0) }
    else:
        # weighted mean/std
        # reshape x to 1 dim
        m = np.average(x, axis=0, weights=weights)
        v = np.sqrt(np.dot(weights, (x - m)**2) / weights.sum())
        ret = {'mean': m, 'std': v}

    # replace zero values
    if len(x.shape) == 1:
        if ret['std'] == 0:
            ret['std'] = 1
    else:
        zero_std = [k for k in xrange(x.shape[1]) if ret['std'][k] < 1e-16]
        for i in zero_std:
            ret['std'][i] = 1.0

    return ret

