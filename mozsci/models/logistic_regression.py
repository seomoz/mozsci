
import numpy as np
import json

class LogisticRegression(object):
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
        return LogisticRegression._sigmoid(x, self.b, self.w)

    def fit(self, x, y, weights=None, **kwargs):
        """Train the model.

        x = (Nobs, nvars)
        y = (Nobs, )  = {0, 1}

        Bias term automatically added

        Returns the loss

        **kwags passed into fmin_l_bfgs_b"""
        from scipy.optimize import fmin_l_bfgs_b

        assert len(y) == x.shape[0]
        assert weights is None or len(weights) == x.shape[0]

        y0 = y == 0
        x0 = x[y0, :]
        x1 = x[~y0, :]

        if weights is None:
            loss_weights = None
        else:
            loss_weights = [weights[y0], weights[~y0]]

        def _loss_for_optimize(params):
            return LogisticRegression._loss_gradient(x0, x1, params[0], params[1:], self.lam, loss_weights)

        params0 = np.zeros(1 + x.shape[1])
        params_opt, loss_opt, info_opt = fmin_l_bfgs_b(_loss_for_optimize, params0, disp=0, **kwargs)
        print("%s funcalls: %s" % (info_opt['task'], info_opt['funcalls']))

        self.b = params_opt[0]
        self.w = params_opt[1:]

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
    def _sigmoid(x, b, w):
        """Return sigma(x) = 1.0 / (1.0 + exp(-x * w - b))
        X = N x (nvars)
        
        Returns a (N, ) array"""
        return np.minimum(np.maximum(1.0 / (1.0 + np.exp(-b - np.sum(w * x, axis=1))), 1.0e-12), 1 - 1.0e-12)

    @staticmethod
    def _loss_gradient(x0, x1, b, w, lam, weights=None):
        """Return loss/gradient function at x.
        x0 = (N0, nvars) numpy array of x where y == 0
        x1 = (N1, nvars) numpy array of x where y == 1

        loss = Logistic loss + 0.5 * lambda * sum(w**2)
        logistic loss =  -sum ( log(sigmoid(x))   y == 1
                                log(1 - sigmoid(x)) if y == 0 )
        weights = if provided an [(N0, ), (N1, )] list of arrays to add in to each
            observation's contribution to error.
            first entry corresponds to x0, second to x1
        """
        nvars = len(w)

        # initialize + regularization term
        loss = 0.5 * lam * np.sum(w ** 2)
        gradient = np.zeros(nvars + 1)               # first position is b
        gradient[1:] = lam * w

        # we need prediction for x
        pred_x_0_1 = [LogisticRegression._sigmoid(x0, b, w), LogisticRegression._sigmoid(x1, b, w)]

        # the log likelihood
        log_like_x_0_1 = [np.log(1.0 - pred_x_0_1[0]),
                          np.log(pred_x_0_1[1])]

        # also need the error for gradient.
        error = [pred_x_0_1[0],
                 pred_x_0_1[1] - 1]

        if weights is None:
            loss += -np.sum(log_like_x_0_1[1]) - np.sum(log_like_x_0_1[0])
            gradient[0] += np.sum(error[0]) + np.sum(error[1])   # * 1 for bias term 
            for k in xrange(nvars):
                gradient[k + 1] += np.sum(error[0] * x0[:, k]) + np.sum(error[1] * x1[:, k])
        else:
            loss += -np.sum(weights[1] * log_like_x_0_1[1]) - np.sum(weights[0] * log_like_x_0_1[0])
            gradient[0] += np.sum(error[0] * weights[0]) + np.sum(error[1] * weights[1])
            for k in xrange(nvars):
                gradient[k + 1] += ( np.sum(weights[0] * error[0] * x0[:, k]) +
                                     np.sum(weights[1] * error[1] * x1[:, k]) )
        return loss, gradient

