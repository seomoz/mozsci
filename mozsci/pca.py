
# linear PCA
import json

import numpy as np
import pylab as plt

from .numpy_util import numpy_decoder, NumpyEncoder

class LinearPCA(object):
    """Linear PCA by SVD"""
    
    def __init__(self, json_map=None):
        """Constructor
        If json_map is provided, then initializes from it"""
        if json_map is None:
            self.mean = None
            self.nvars = None
            self.eigval = None
            self.eigvec = None
        else:
            j = json.loads(json_map, object_hook=numpy_decoder)
            self.mean = j['mean']
            self.nvars = j['nvars']
            self.eigval = j['eigval']
            self.eigvec = j['eigvec']



    def train(self, data, fignum=None):
        """Train the PCA.  data is an (nobs, nvars) numpy array
        If fignum is not None, then plot the eigen values in the figure
        
        Returns nothing."""
        assert isinstance(data, np.ndarray) and data.ndim == 2 
        self.nvars = data.shape[1]
        self.mean = np.mean(data, 0)

        # do SVD of the data
        corr = np.cov((data - self.mean).T)
        (eigval, eigvec) = np.linalg.eig(corr)

        # sort eigenvalues, eigen vectors into ascending order
        sortindex = (-1.0 * eigval).argsort()
        eigval = eigval[sortindex]
        eigvec = eigvec[:, sortindex]

        self.eigval = eigval
        self.eigvec = eigvec

        # plot eigenvalues
        if fignum is not None:
            eigval_sum = self._compute_percent_explained()

            fig = plt.figure(fignum)
            fig.clf()
            plt.plot(eigval_sum, 'bx')
            plt.title("Eigenvalues for PCA")
            fig.show()

    def _compute_percent_explained(self):
        """Computes percent explained from self.eigval"""
        eigval_sum = self.eigval.cumsum()
        eigval_cum_sum = eigval_sum / np.float(eigval_sum[-1])
        percent_explain = np.hstack((eigval_cum_sum[0], eigval_cum_sum[1:] - eigval_cum_sum[0:-1]))
        return percent_explain

    def plot_eigvec(self, neig, fignum):
        """Plots the first neig eigenvectors for figure fignum"""
        fig = plt.figure(fignum)
        fig.clf()
        pct_explain = self._compute_percent_explained()
        for k in xrange(neig):
            plt.plot(self.eigvec[:, k], label=str(k) + " " + str(round(pct_explain[k] * 100)))
        plt.legend()
        fig.show()


    def project(self, data, n):
        """Given the data, project onto the first n principle components.
        data must have the same number of variables as the data used
        in training.

        data is a (nobs, nvars) numpy array
        return is a (nobs, n) numpy array, the projection onto the PCA
        
        Note: the mean (self.mean) is removed from the data before
        projection so that the full projection is
            self.mean + SUM_k (projection_k * PC_k)"""
        assert data.ndim == 2 and data.shape[1] == self.nvars and n > 0 and n <= self.nvars
        return np.dot((data - self.mean), self.eigvec[:, 0:n])

    def truncate(self, data, n):
        """Truncate the data to the n PCs.
        This projects on the first n PCs, then reconstructs data.

        data is a (nobs, nvars) numpy array
        return is a (nobs, nvars) numpy array"""
        return self.mean + np.dot(self.project(data, n), self.eigvec[:, 0:n].T)

    def to_json(self):
        """Returns a json string with the PCA"""
        j = {}
        j['eigval'] = self.eigval
        j['eigvec'] = self.eigvec
        j['mean'] = self.mean
        j['nvars'] = self.nvars
        return json.dumps(j, cls=NumpyEncoder)


