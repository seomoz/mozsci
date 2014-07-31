"""Input feature manipulation, including normalizations"""

import numpy as np

from sklearn.preprocessing import StandardScaler

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


class IdentityTransformer(object):
    '''
    Identity transformer that implements sklearn Transformer API
    '''
    def transform(self, X, *args, **kwargs):
        return X

    def fit(self, X, *args, **kwargs):
        pass


class LogScaledTransformer(StandardScaler):
    def __init__(self, offset=0.0, **kwargs):
        '''
        Take log(X+offset) then apply mean-std scaling.
        **kwargs: passed into StandardScaler.__init__

        we ignore the copy options for convenience
        '''
        super(LogScaledTransformer, self).__init__(**kwargs)
        self._offset = offset

    def _log(self, X):
        return np.log(X + self._offset)

    def fit(self, X, *args, **kwargs):
        XX = self._log(X)
        return super(LogScaledTransformer, self).fit(XX, *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        XX = self._log(X)
        return super(LogScaledTransformer, self).transform(
            XX, *args, **kwargs)

    def inverse_transform(self, X, *args, **kwargs):
        XX = super(LogScaledTransformer, self).inverse_transform(
            X, *args, **kwargs)
        return np.exp(XX) - self._offset


class BucketTransformer(object):
    '''
    Transform a float to a categorical variable and represent as
        1-in-k encoding.
    '''
    def __init__(self, bin_edges):
        '''
        bin_edges: edges for the len(bin_edges) + 1 bins.  They are:

        bin_edges = [x0, x1, ..., xn]
            x <= x0
            x0 < x <= x1
            ...
            xn < x
        '''
        from sklearn.preprocessing import Binarizer
        self._binarizers = [Binarizer(threshold=-np.inf)]
        self._binarizers.extend(
            [Binarizer(threshold=edge) for edge in bin_edges])
        self._nbins = len(self._binarizers)

    def fit(self, *args, **kwargs):
        pass

    def transform(self, X):
        '''
        X = len N vector
        return (N, nbins) matrix with 1-in-k encoding
        '''
        assert(len(X.shape) == 1)

        ret = np.zeros((len(X), self._nbins))
        for k, binarizer in enumerate(self._binarizers):
            ret[:, k] = binarizer.transform(X)

        # since binarizer is 0-1 for whether X is less then the threshold
        # we need the last 1 in each column, e.g.
        #
        # [1, 1, 0, 0] we change to [0, 1, 0, 0]
        # can get the value by subtracting the previous column
        for k in xrange(self._nbins-1):
            ret[:, k] = ret[:, k] - ret[:, k+1]
        return ret

