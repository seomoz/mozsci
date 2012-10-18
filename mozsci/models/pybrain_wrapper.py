
# a wrapper for pybrain networks
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet
import numpy as np

class PyBrainNNError(Exception):
    pass

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
        loss /= X.shape[0]

        # gradient = 2 * (predy - y) * dnet/dparam
        gradient = 2 * np.dot(err.reshape(1,-1), net_grad) + self.lam * self.net.params[:]
        gradient /= X.shape[0]

        return loss, gradient



    def fit(self, X, y):
        """Train the network"""
        # batch gradient descent
        err = 1e20
        err_delta = 1e20
        iteration = 0
        nerror_decrease = 0
        while(err_delta > self.errtol and iteration < self.maxiterations):
            err_new, grad = self._r2_loss_gradient(X, y)
            err_delta = abs(err_new - err)

            # update the params if error decreased
            # otherwise, decrease learning rate
            if err_new <= err:
                self.net.params[:] = self.net.params[:] - self.learning_rate * grad
                err = err_new
                iteration += 1
                print("Iteration %s, error=%s" % (iteration, err))
            else:
                # error increased.  we must have too large of a learning rate
                # decrease it and try again
                self.learning_rate = self.learning_rate * 0.95
                nerror_decrease += 1
                print("Iteration %s, decreased learning rate to %s, decrease #" % (iteration, self.learning_rate, nerror_decrease))

                # perturb the parameters a little
                max_param = self.params[:].max()
                self.params[:] = (np.random.rand(self._nparams) - 0.5) * 2 * max_param * 0.05 + self.params[:]
                if nerror_decrease == 100:
                    raise PyBrainNNError("Decreased learning rate 100 times and error still increasing")


    def predict(self, X):
        """Do predictions"""
        ypred = np.zeros(X.shape[0])
        for k in xrange(X.shape[0]):
            self.net.reset()
            ypred[k] = self.net.activate(X[k, :])
        return ypred

