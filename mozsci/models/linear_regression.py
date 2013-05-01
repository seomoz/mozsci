
import numpy as np
from scipy.optimize import fmin_bfgs
import json

class LinearRegression(object):
    def __init__(self, lam=1.0):
        """lam = regularization parameter"""
        self.lam = lam

        # these are set in fit
        self.b = None  # float
        self.w = None  # (nvars, ) array

    def predict(self, x):
        """Make a prediction.
        Return P(y == 1 | x)

        x = (Nobs, nvars)
        """
        return np.sum(self.w * x, axis=1) + self.b

    def fit(self, x, yy, weights=None):
        """Train the model.

        x = (Nobs, nvars)
        y = (Nobs, )

        Bias term automatically added

        Returns the loss"""
        # transform y to vector
        if len(yy.shape) > 1:
            assert len(yy.shape) == 2 and yy.shape[1] == 1
            y = yy.reshape(-1, )
        else:
            y = yy

        def _loss_for_optimize(params):
            return LinearRegression._loss(x, y, params[0], params[1:], self.lam, weights)
        def _gradient_for_optimize(params):
            return LinearRegression._gradient_loss(x, y, params[0], params[1:], self.lam, weights)

        params_opt = fmin_bfgs(_loss_for_optimize, np.zeros(1 + x.shape[1]), fprime=_gradient_for_optimize, maxiter=200)

        self.b = params_opt[0]
        self.w = params_opt[1:]

        return _loss_for_optimize(params_opt)

    def save_model(self, model_file):
        """Serialize model to model_file"""
        m = {'b':self.b,
            'w':self.w.tolist()}

        with open(model_file, 'w') as f:
            json.dump(m, f)

    @classmethod
    def load_model(cls, model_file):
        '''If a string is provided, it's assumed to be a path to a file
        containing a JSON blob describing the model. Otherwise, it should
        be a dictionary representing the model'''
        if isinstance(model_file, basestring):
            params = json.load(open(model_file, 'r'))
        else:
            params = model_file
        ret = cls()
        ret.b = float(params['b'])
        ret.w = np.array(params['w'])
        return ret

    @staticmethod
    def _loss(x, y, b, w, lam, weights=None):
        """Return loss function at x.
        loss = sum_squared loss + 0.5 * lambda * sum(w**2)
        weights = if provided an (N, ) list of weights
        """
        loss = 0.5 * lam * np.sum(w ** 2)
        if weights is None:
            loss += np.sum((np.sum(w * x, axis=1) + b - y) ** 2)
        else:
            loss += np.sum(weights * (np.sum(w * x, axis=1) + b - y) ** 2)
        return loss

    @staticmethod
    def _gradient_loss(x, y, b, w, lam, weights=None):
        """Return the gradient of the loss.

           x0 = (N, nvars) numpy array of x
           y = prediction

           gradient = loss + self.lam * w

            weights = if provided an (N, ) array to add in to each
        """
        nvars = len(w)
        gradient = np.zeros(nvars + 1)               # first position is b
        gradient[1:] = lam * w

        # need sum(f(x) - y) * x for all variables
        error = np.sum(w * x, axis=1) + b - y
        if weights is None:
            gradient[0] = np.sum(error)   # * 1 for bias term
            for k in xrange(nvars):
                gradient[k + 1] += np.sum(error * x[:, k])
        else:
            gradient[0] = np.sum(error * weights)   # * 1 for bias term
            for k in xrange(nvars):
                gradient[k + 1] += np.sum(weights * error * x[:, k])

        gradient *= 2

        return gradient

