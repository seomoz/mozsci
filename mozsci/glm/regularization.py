from __future__ import absolute_import
import numpy as np
from six.moves import range

class RegularizationBase(object):
    """
    Base class of all the regularization methods.
    Super classes can provide gradient and Hessian methods.
    """

    def __init__(self):
        pass

    def eval(self, x):
        pass

class NullRegularization(RegularizationBase):
    """
    This is a null regularization, ie. 0 regularization.
    """
    def eval(self, x):
        return 0

    def eval_gradient(self, x):
        return 0

    def eval_hessian(self, x):
        return 0

class RidgeRegularization(RegularizationBase):
    """
    Ridge regularization.
    It's lam/2.0 * ||x|| ** 2
    This regularization does not penalize the constant term. The constant term is supposed to be the last term.
    """

    def __init__(self, lam):
        self.lam = lam

    def eval(self, x):
        return 0.5 * self.lam * np.inner(x[:-1], x[:-1])

    def eval_gradient(self, x):
        tmp = self.lam * x
        tmp[-1] = 0.0

        # return self.lam * x
        return tmp

    def eval_hessian(self, x):
        hessian = self.lam * np.identity(x.shape[0])
        hessian[-1, -1] = 0
        return hessian

class RidgeRegularizationAll(RegularizationBase):
    """
    Ridge regularization.
    It's lam/2.0 * ||x|| ** 2
    """

    def __init__(self, lam):
        self.lam = lam

    def eval(self, x):
        return 0.5 * self.lam * np.inner(x, x)

    def eval_gradient(self, x):
        return self.lam * x

    def eval_hessian(self, x):
        return self.lam * np.identity(x.shape[0])


class RidgeRegularizationChosen(RegularizationBase):
    """
    Ridge regularization on chosen terms.
    It's lam/2.0 * ||x|| ** 2
    """

    def __init__(self, lam, dim, free_list=[]):
        self.lam = lam

        # this is the indices that will be penalized/regulated.
        self.index = list(set(range(dim)) - set(free_list))
        self.free = free_list

    def eval(self, x):
        xx = x[self.index]
        return 0.5 * self.lam * np.inner(xx, xx)

    def eval_gradient(self, x):
        grad = self.lam * x
        grad[self.free] = 0.0

        return grad

    def eval_hessian(self, x):
        hessian = self.lam * np.identity(x.shape[0])
        hessian[self.free, self.free] = 0.0
        return hessian



