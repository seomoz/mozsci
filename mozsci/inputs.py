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
    def set_params(self, params):
        pass

    def get_params(self):
        return None

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

