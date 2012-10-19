
# a wrapper for pybrain networks
from pybrain.tools.shortcuts import buildNetwork
import numpy as np
import json

class PyBrainNNError(Exception):
    pass

class PyBrainNN(object):
    """Encapsulates a pybrain network and adds L2 regularization.

    Exposes the scikit-learn interface, trains by batch gradient descent"""
    def __init__(self, errtol=1.0e-10, maxiterations=200, learning_rate=1.0, lam=1.0, args=(), kwargs={}):
        """Initialize the network.

        errtol = stopping tolerance
        maxiterations = only do this many iterations, maximum
        learning rate = for gradient descent
        lam = regularization parameter

        All input passed down to buildNetwork"""
        self._training_params = {'errtol':errtol, 'maxiterations':maxiterations, 'learning_rate':learning_rate, 'lam':lam}
        self.net = buildNetwork(*args, **kwargs)
        self._nparams = len(self.net.params)

        # we'll stash these away for saving the network
        self._args = args
        self._kwargs = kwargs

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
        loss = np.sum(err**2) + 0.5 * self._training_params['lam'] * np.sum(self.net.params[:] ** 2)
        loss /= X.shape[0]

        # gradient = 2 * (predy - y) * dnet/dparam
        gradient = 2 * np.dot(err.reshape(1,-1), net_grad) + self._training_params['lam'] * self.net.params[:]
        gradient /= X.shape[0]

        return loss, gradient


    def save_model(self, fileout):
        """Saves the model to the file.
        
        fileout is a string name of the file, or implements write method"""

        # we'll make a json blob to save
        model_json = {}
        model_json['params'] = self.net.params[:].tolist()
        model_json['args'] = self._args
        model_json['kwargs'] = self._kwargs
        model_json['training_params'] = self._training_params

        # save to the file
        if isinstance(fileout, basestring):
            with open(fileout, 'w') as f:
                json.dump(model_json, f)
        else:
            json.dump(model_json, fileout)

    @classmethod
    def load_model(cls, model_json):
        """Load in the model

        model_json is either a dictionary with the required model attributes,
        or a file name with this dictionary deserialized into a json string"""
        if isinstance(model_json, basestring):
            with open(filename, 'r') as f:
                model_json = json.load(f)

        # need keyword args to be strings in Python 2.6 :-/
        # conver them - assumes model_json has no more then 1 nested dict
        model_json_str = {}
        for k,v in model_json.iteritems():
            if isinstance(v, dict):
                model_json_str[str(k)] = {}
                for kk, vv in v.iteritems():
                    model_json_str[str(k)][str(kk)] = vv
            else:
                model_json_str[str(k)] = v

        model = cls(args=model_json_str['args'], kwargs=model_json_str['kwargs'], **model_json_str['training_params'])
        model.net.params[:] = model_json['params']
        return model


    def fit(self, X, y):
        """Train the network"""
        # batch gradient descent

        # compute the initial error
        err, grad = self._r2_loss_gradient(X, y)

        err_delta = 1e20
        iteration = 0
        nerror_decrease = 0

        while(err_delta > self._training_params['errtol'] and iteration < self._training_params['maxiterations']):

            # take a gradient step
            params_old = self.net.params.copy()
            self.net.params[:] = self.net.params[:] - self._training_params['learning_rate'] * grad

            # check the error
            err_new, grad_new = self._r2_loss_gradient(X, y)
            if err_new <= err:
                # error decreased
                # keep the parameter updates, set for next iteration
                err_delta = abs(err_new - err)
                err = err_new
                grad = grad_new
                iteration += 1
                print("Iteration %s, error=%s" % (iteration, err))

            else:
                # error increased.  we must have too large of a learning rate
                # decrease it and try again
                self._training_params['learning_rate'] = self._training_params['learning_rate'] * 0.95
                nerror_decrease += 1
                self.net.params[:] = params_old
                
                print("Iteration %s, decreased learning rate to %s, decrease # %s" % (iteration, self._training_params['learning_rate'], nerror_decrease))

                # perturb the parameters a little
                #max_param = self.net.params[:].max()
                #self.net.params[:] = (np.random.rand(self._nparams) - 0.5) * 2 * max_param * 0.05 + self.net.params[:]
                if nerror_decrease == 100:
                    raise PyBrainNNError("Decreased learning rate 100 times and error still increasing")


    def predict(self, X):
        """Do predictions"""
        ypred = np.zeros(X.shape[0])
        for k in xrange(X.shape[0]):
            self.net.reset()
            ypred[k] = self.net.activate(X[k, :])
        return ypred

