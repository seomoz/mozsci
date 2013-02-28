import numpy as np
import simplejson as json

from . import regularization
from . import prob_distributions
from scipy import optimize

"""
This module defines a series of simplified glm functions. They are simplified because they use a much simpler way to
link the probability distribution parameters and the observed feature variables.
"""

class SimplifiedGlmBase(object):
    """
    This is the base class of the simplified glm classes.
    Each super class must provide both fit and predict methods.
    """

    def __init__(self, lam=0.1, regular=None, seed=None, likelihood=None, initialize_params=None, param_len=None,
                 maxiter=None):
        """
        :param lam: the parameter for regularization.
        :param regular:
        :param seed:
        :param likelihood:
        :param initialize_params: the method to initialize the parameter array. It takes one parameter as the length
               of the params array. See the definition of random_initialize_params(len).
        :param maxiter: the number of iterations that can run when do the 'fitting' of the model. It controls the
                        time spent on optimization routine.
        """

        if lam is not None:
            self.regularization = regularization.RidgeRegularization(lam)
        else:
            if regular is None:
                self.regularization = regularization.RidgeRegularization(0.1)
            else:
                self.regularization = regular

        self.params = None  ## The last number in this 1-d array is for the constant (1.0) term.
        self.likelihood = likelihood

        if seed is not None:
            np.random.seed(seed)
        else:           ## Likely I will delete these two lines in the future to let the numpy use its own.
            np.random.seed(4559)

        if initialize_params is None:
            self.initialize_params = np.zeros
        else:
            self.initialize_params = initialize_params

        self.param_len = param_len
        self.maxiter = maxiter  ## To control the optimization routine.

    def get_eval(self, features, y):
        """
        A wrapper of the likelihood and regularization terms for easy use with scipy's optimization routines.
        :param features:
        :param y:
        :return: the value of the objective function which is -loglikelihood + regularization. We want to
                 minimize it.
        """

        def func(beta):
            return -self.likelihood.eval(beta, features, y) + self.regularization.eval(beta)
        return func

    def get_gradient(self, features, y):
        """
        A wrapper of the likelihood and regularization terms for easy use with scipy's optimization routines.
        :param features:
        :param y:
        :return:
        """

        def func(beta):
            return -self.likelihood.eval_gradient(beta, features, y) + self.regularization.eval_gradient(beta)
        return func

    def get_hessian(self, features, y):
        """
        A wrapper of the likelihood and regularization terms for easy use with scipy's optimization routines.
        :param features:
        :param y:
        :return:
        """

        def func(beta):
            return -self.likelihood.eval_hessian(beta, features, y) + self.regularization.eval_hessian(beta)
        return func


    def fit(self, x, y):
        """
        training the model.
        :param x: the design matrix. It doesn't need to have the constant column, because we are adding one.
        :param y: the observed independent variables.
        :return:
        """
        # add the constant column as the last column
        features = np.c_[x, np.ones(x.shape[0])]

        # setup the param length. This is usually the number of features plus 1 for the constant term. but there are exceptions, such as negative binomial.
        if self.param_len is None:
            self.param_len = features.shape[1]

        initial_params = self.initialize_params(self.param_len)

        eval_func = self.get_eval(features, y)
        eval_gradient_func = self.get_gradient(features, y)

        # I have tried Newton, Secant, Conjugate Gradient etc to see which one is more robust. Speed is less important.
        # http://scipy-lectures.github.com/advanced/mathematical_optimization/index.html
        # eval_hessian_func = self.get_hessian(features, y)

        self.params = optimize.fmin_bfgs(eval_func, initial_params, fprime=eval_gradient_func, maxiter=self.maxiter)

    def predict(self, x):
        """
        This predict actually returns the the inverse of the expectation.
        :param x: design matrix
        :return:
        """
        features = np.c_[x, np.ones(x.shape[0])]
        return np.inner(features, self.params)

    def save_model(self, model_file):
        """Serialize model to model_file"""
        m = {'params':self.params.tolist()}

        with open(model_file, 'w') as f:
            json.dump(m, f)

    @classmethod
    def load_model(cls, model_file):
        """
        load the model from a file or a json block.
        """

        if isinstance(model_file, basestring):
            params = json.load(open(model_file, 'r'))
        else:
            params = model_file
        ret = cls()
        ret.params = np.array(params['params'])
        return ret

class PoissonRegression(SimplifiedGlmBase):
    """
    prob dist: Poisson.
    lambda is a linear function of feature variables.
    Expected value is exp(w * x) where x * x is the inner product.
    """
    def __init__(self, *args, **kw):

        super(PoissonRegression, self).__init__(likelihood=prob_distributions.Poisson(), *args, **kw)

class NegativeBinomialWithKstarRegression(SimplifiedGlmBase):
    """
    prob dist: Poisson.
    lambda is a linear function of feature variables.
    Expected value is exp(w * x) where x * x is the inner product.
    """
    def __init__(self, beta_k_len, initial_k_star=9, regular=None, lam=1.0, *args, **kw):
        """
        :param initial_k_star:  the initial value for k*, ie. log(k).
        """

        if regular is None:
            print 'lam is --from nb inside ', lam
            ## The first entry is the k_star, ie. the number of failures in negative binomial.
            ## The last entry is the constant term in the linear regression.
            regular = regularization.RidgeRegularizationChosen(lam, dim=beta_k_len, free_list=[0, beta_k_len - 1])

        self.initial_k_star = initial_k_star

        super(NegativeBinomialWithKstarRegression, self).__init__(lam=None, regular=regular,
            likelihood=prob_distributions.NegativeBinomialWithKstar(),
            initialize_params=self.initialize_params_withk,
            param_len=beta_k_len, *args, **kw)

    def initialize_params_withk(self, cnt):
        """
        To return a function to initialize the beta and initial k*.
        cnt should be one more of all the features.
        """
        params = np.zeros(cnt)
        params[0] = self.initial_k_star

        return params

    def predict(self, x):
        """
        This overrides the base class's predict.
        :param x: design matrix
        :return: predicted y.
        """
        # The first entry of the params is the k*, ie. log(k)
        params = self.params[1:]

        features = np.c_[x, np.ones(x.shape[0])]
        return np.inner(features, params)

class ExponentialGlm(SimplifiedGlmBase):
    """
    prob dist: Exponential.
    lambda is a linear function of feature variables.
    Expected value is exp(w * x) where w * x is actually the inner product of (w, x)
    """

    def __init__(self, *args, **kw):

        super(ExponentialGlm, self).__init__(likelihood=prob_distributions.Exponential(), *args, **kw)

def random_initialize_params(array_len):
    """
    Create an 1-d array that is uniformly randomly chosen in [-0.5, 0.5]
    :param array_len: how long is the array.
    :return: the numpy array.
    """
    return np.random.rand(array_len) - 0.5


