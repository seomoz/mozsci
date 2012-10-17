
# a wrapper for pybrain networks
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np

class PyBrainNN(object):

    def __init__(self, errtol=1.0e-10, maxiterations=200, learning_rate=1.0, lam=1.0, args=(), kwargs={}):
        """Initialize the network.

        errtol = stopping tolerance
        maxiterations = only do this many iterations, maximum
        learning rate = for gradient descent
        lam = regularization parameter

        All input passed down to buildNetwork"""
        self.errtol = errtol
        self.maxiterations = maxiterations
        self.learning_rate = learning_rate
        self.lam = lam
        self.net = buildNetwork(*args, **kwargs)
        self._nparams = len(self.net.params)

    def _eval_gradient(self, X):
        """evaluate and compute gradient of network wrt parameters"""
        nresult = X.shape[0]
        pred = np.zeros(nresult)
        grad = np.zeros((nresult, self._nparams))
        for k in xrange(nresult):
            self.net.reset()
            self.net.resetDerivatives()
            pred[k] = self.net.activate(X[k, :])
            dummy = self.net.backActivate(1.0)
            grad[k, :] = self.net.derivs
        return (pred, grad)

    def _r2_loss_gradient(self, X, y):
        """Compute the sum-squared loss and gradient over the data set"""
        gradient = np.zeros(self._nparams)
        net_pred, net_grad = self._eval_gradient(X)

        # loss is SUM(predy - y)**2 over the data set
        err = net_pred - y
        loss = np.sum(err**2) + 0.5 * self.lam * np.sum(self.net.params[:] ** 2)

        # gradient = 2 * (predy - y) * dnet/dparam
        gradient = 2 * np.dot(err.reshape(1,-1), net_grad) + self.lam * self.net.params[:]

        return loss, gradient



    def fit(self, X, y):
        """Train the network"""
        # batch gradient descent
        err = 1e20
        err_delta = 1e20
        iteration = 0
        while(err_delta > self.errtol and iteration < self.maxiterations):
            err_new, grad = self._r2_loss_gradient(X, y)
            err_delta = abs(err_new - err)
            err = err_new
            iteration += 1

            # update the params
            self.net.params[:] = self.net.params[:] - self.learning_rate * grad
            print("Iteration %s, error=%s" % (iteration, err))


    def predict(self, X):
        """Do predictions"""
        ypred = np.zeros(X.shape[0])
        for k in xrange(X.shape[0]):
            self.net.reset()
            ypred[k] = self.net.activate(X[k, :])
        return ypred

