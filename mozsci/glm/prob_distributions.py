"""
This module provides all the probability distributions that the simplified generalized models supports.
A better name might be likelihood. We will provide the eval, eval_gradient, and eval_hessian for the
log likelihood here. Note, we do not add the negative here. That should be done by the caller.
"""
from __future__ import absolute_import
from __future__ import print_function

### Attention, there is no special treatment of the constant column here. So before call any method here,
### add one columns of 1's to the feature matrix, ex, np.c_[features, np.ones(features.shape[0])]

import numpy as np

class GlmProbDistBase(object):
    """
    The base class of the probability distributions.
    """

    def __init__(self):
        pass

    def eval(self, beta, features, y):
        """
        This method returns the log likelihood of the variables. Constants will be omitted because the goal of
        this evaluation is to maximize the log likelihood. So, this method might return positive numbers.
        """
        pass

    def eval_gradient(self, beta, features, y):
        pass

    def eval_hessian(self, beta, features, y):
        pass

    def get_inverse_link(self):
        """
        Get the inverse of the link function. The caller can use it to calculate the expected value from
        the linear predictors.
        """
        pass

class Poisson(GlmProbDistBase):
    """
    Poisson regression.
    """
    def eval(self, beta, features, y):
        """
        return the log likelihood, with features
        """

        log_miu = np.dot(features, beta)
        log_miu = np.minimum(log_miu, 5)
        tmp = np.sum(log_miu * y - np.exp(log_miu))

        if np.isinf(tmp):
            print('WARNING -- Log likelihood got inf value. It has been replaced by float.max. ')
            print('max of y * log miu', np.max(y * log_miu))
            print('max of  miu', np.max(np.exp(log_miu)))
            print('max of y ', max(y))

            return np.finfo(np.float).max
        else:
            return tmp

    def eval_gradient(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        This is the gradient against beta_k. beta_k[0] = k* which is the log of k.
        This is a faster version, compared with the eval_gradient_bk
        :param beta_k: one single array of k* and beta
        :param features:
        :param y: observed variable.
        :return:
        """
        # setup the values we are going to need.
        log_miu = np.dot(features, beta)
        # prevent overflows
        log_miu = np.minimum(log_miu, 5)
        miu = np.exp(log_miu)
        grad_tmp = y - miu

        gradient = np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient)):
            print('Warning--The grad_tmp has nan', gradient)

        return gradient

    def get_inverse_link(self):
        return np.exp

class Exponential(GlmProbDistBase):
    """
    The exponential probability distribution. The parameter lambda is the inner product of beta and x.
    This exponential uses a different link function. log(x). This solves the non-positive problem we have
    in Expontial class.
    """

    def eval(self, beta, features, y):
        """
        return the log likelihood
        theta = beta * feature.
        """

        log_miu = np.dot(features, beta)
        tmp = -np.sum(log_miu + y * np.exp(-log_miu))

        if np.isinf(tmp):
            print('WARNING -- Log likelihood got inf value. It has been replaced by float.max. ')
            print('max of log miu', np.max(log_miu))
            print('max of  y / miu', np.max(y * np.exp(-log_miu)))
            print('max of y ', max(y))

            return np.finfo(np.float).max
        else:
            return tmp

    def eval_gradient(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        """
        # setup the values we are going to need.
        log_miu = np.dot(features, beta)
        grad_tmp = 1.0 - y * np.exp(-log_miu)

        gradient = -np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient)):
            print('Warning--The grad_tmp has nan', gradient)

        return gradient

    def get_inverse_link(self):
        return np.exp

class NegativeBinomialWithKstar(GlmProbDistBase):
    """
    Negative Binomial regression.
    Parameter k is fixed.
    """
    def eval(self, beta_k, features, y):
        """
        return the log likelihood, with feature feature
        theta = beta * feature.
        Attention: We omit the ln((y-1)!) in the loglikelihood, because our goal is to optimize the loglikelihood.
        beta_k[0] = k* which is the log of k.
        """
        beta = beta_k[1:]

        # underflow in some special cases.
        if beta_k[0] < -720.0:
            beta_k[0] = -720.0

        k = np.exp(beta_k[0])           ## exp(k*).
        ln_exp_k_star = beta_k[0]       ## ln(e^(k*)). It's actually k*, ie. beta_k[0]

        max_y = int(y.max())
        subsum_y = np.log(np.arange(max_y) + k).cumsum()
        log_miu = np.dot(features, beta)

        # log( 1 + exp( k* - log(miu)))
        log_1_plus_sth = np.log(1.0 + np.exp(beta_k[0] - log_miu))
        log_1_plus_sth[beta_k[0] - log_miu > 50] = beta_k[0] - log_miu[beta_k[0] - log_miu > 50]

        subsum = subsum_y[y.astype(np.int) - 1]
        subsum[y.astype(np.int) == 0] = 0.0

        tmp = np.sum(subsum + k * ln_exp_k_star + y * log_miu) - np.sum((k + y) * (log_miu + log_1_plus_sth))

        if np.isinf(tmp):
            print('WARNING -- Log likelihood got inf value. It has been replaced by float.max. ')
            print('max of subsum', np.max(subsum))
            print('max of y * log miu', np.max(y * log_miu))
            print('max of (k+y) * log miu and k',  np.max((k + y) * (log_miu + log_1_plus_sth)))
            print('max of log miu and log 1 puls sth', np.max(log_miu), np.max(log_1_plus_sth))
            print('max of y ', max(y))
            print('value of  exp and k', k * ln_exp_k_star)

            return np.finfo(np.float).max
        else:
            return tmp

    def eval_gradient(self, beta_k, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        This is the gradient against beta_k. beta_k[0] = k* which is the log of k.
        This is a faster version, compared with the eval_gradient_bk
        :param beta_k: one single array of k* and beta
        :param features:
        :param y: observed variable.
        :return: the gradient of the log likelihood
        """
        # setup the values we are going to need.
        beta = beta_k[1:]

        if beta_k[0] < -720.0:  # handling underflow.
            beta_k[0] = -720.0

        k = np.exp(beta_k[0])           ## exp(k*).

        log_miu = np.dot(features, beta)
        log_1_plus_sth = np.log(1.0 + np.exp(beta_k[0] - log_miu))
        log_1_plus_sth[beta_k[0] - log_miu > 50] = beta_k[0] - log_miu[beta_k[0] - log_miu > 50]

        miu = np.exp(log_miu)
        miu[np.isinf(miu + k)] = np.finfo(np.float).max - 1.5 * k

        # gradient of beta
        grad_tmp = (y - miu) / (miu + k)
        # test of nan in the gradient calculation.
        if np.isnan(np.sum(grad_tmp)):
            if np.isnan(np.sum(miu)):
                print('The miu has nan', miu)
            print('min of miu + k is ', np.min(miu + k))
            print('max of miu + k is ', np.max(miu + k))
            print('min of y - miu is ', np.min(y - miu))
            print('max of y - miu is ', np.max(y - miu))
            print('The grad_tmp has nan', grad_tmp)

        gradient_beta = k * np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient_beta)):
            print('The grad_tmp has nan', gradient_beta)

        # derivative of k*
        max_y = int(y.max())
        subsum_y = (1.0 / (np.arange(max_y) + k)).cumsum()
        subsum = subsum_y[y.astype(np.int) - 1]
        subsum[y.astype(np.int) == 0] = 0.0

        derivative_k = np.sum(subsum + 1.0 + beta_k[0] - (k + y)/(k + miu) - (log_miu + log_1_plus_sth))

        if np.isinf(derivative_k):
            print('WARNING -- Derivative of kstar got inf value. It has been replaced by float.max. ')
            derivative_k = np.finfo(np.float).max

        # Assemble them together!
        gradient = np.zeros(beta_k.shape[0])
        gradient[0] = k * derivative_k
        gradient[1:] = gradient_beta

        return gradient

    def get_inverse_link(self):
        return np.exp


