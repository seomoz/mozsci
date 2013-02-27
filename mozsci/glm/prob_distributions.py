"""
This module provides all the probability distributions that the simplified generalized models supports.
A better name might be likelihood. We will provide the eval, eval_gradient, and eval_hessian for the
log likelihood here. Note, we do not add the negative here. That should be done by the caller.
"""

### Attention, there is no special treatment of the constant column here. So before call any method here,
### add one columns of 1's to the feature matrix, ex, np.c_[features, np.ones(features.shape[0])]

## Todo: support weights.

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

class  Poisson(GlmProbDistBase):
    """
    Poisson regression.
    """
    def eval(self, beta, features, y):
        # def eval_bk(self, beta_k, features, y):
        """
        return the log likelihood, with feature feature
        theta = beta * feature.
        Attention: We omit the ln((y-1)!) in the loglikelihood, because our goal is to optimize the loglikelihood.
        beta_k[0] = k* which is the log of k.
        :param beta:
        :param feature:
        :param y:
        :return:
        """

        log_miu = np.dot(features, beta)
        tmp = np.sum(log_miu * y - np.exp(log_miu))

        if np.isinf(tmp):
            print 'WARNING -- Log likelihood got inf value. It has been replaced by float.max. '
            print 'max of y * log miu', np.max(y * log_miu)
            print 'max of  miu', np.max(np.exp(log_miu))
            print 'max of y ', max(y)

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
        miu = np.exp(log_miu)
        grad_tmp = y - miu

        gradient = np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient)):
            print 'Warning--The grad_tmp has nan', gradient

        return gradient


    def eval_bk(self, beta, features, y):
        """
        return the log likelihood, with feature feature
        theta = beta * feature.
        Attention: We omit the factorial of y in the loglikelihood.
        Compared with the eval version, this one is slow but stable.
        :param beta:
        :param feature:
        :param y:
        :return:
        """
        sum = 0.0

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            sum += -np.exp(inner) + y[i] * inner

        return sum

    def eval_gradient_bk(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        :param beta:
        :param features:
        :param x:
        :return:
        """

        gradient = np.zeros(beta.shape[0])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            gradient += -features[i] * np.exp(inner) + y[i]*features[i]

        return gradient

    def eval_hessian(self, beta, features, y):
        """
        return the Hessian matrix of beta at y with feature features.
        dim(beta) = 1. Its length is the dimension of feature space.
        dim(features) = 2. design matrix.
        dim(y) = 1, the observed variables.

        The hessian is actually not using any value in y.
        :return: the hessian
        """

        hessian = np.zeros([beta.shape[0], beta.shape[0]])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            exp_inner_sq = -np.exp(2 * inner)
            for j in range(beta.shape[0]):
                for k in range(beta.shape[0]):
                    hessian[j][k] += features[i][j] * features[i][k] * exp_inner_sq

        return hessian

    def get_inverse_link(self):
        return np.exp


class Exponential(GlmProbDistBase):
    """
    Attention: This one uses the link function of 1/x. This can easily cause problems when x is negative,
    because the expectation should be always positive.
    The exponential probability distribution. The parameter lambda is the inner product of beta and x.
    """

    def eval(self, beta, features, y):
        """
        return the pdf at x, with feature feature
        theta = beta * feature.
        :param beta:
        :param feature:
        :param y:
        :return:
        """
        sum = 0.0

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            sum += np.log(inner) - y[i] * inner

        return sum

    def eval_gradient(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        :param beta:
        :param features:
        :param x:
        :return:
        """

        gradient = np.zeros(beta.shape[0])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            gradient += features[i]/inner - y[i]*features[i]

        return gradient

    def eval_hessian(self, beta, features, y):
        """
        return the Hessian matrix of beta at y with feature features.
        dim(beta) = 1. Its length is the dimension of feature space.
        dim(features) = 2. design matrix.
        dim(y) = 1, the observed variables.

        The hessian is actually not using any value in y.
        :return: the hessian
        """

        hessian = np.zeros([beta.shape[0], beta.shape[0]])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            neg_inner_sq = - inner * inner
            for j in range(beta.shape[0]):
                for k in range(beta.shape[0]):
                    hessian[j][k] += features[i][j] * features[i][k] / neg_inner_sq

        return hessian

    def get_inverse_link(self):
        return lambda x: 1.0/x


class Exponential2(GlmProbDistBase):
    """
    The exponential probability distribution. The parameter lambda is the inner product of beta and x.
    This exponential uses a different link function. log(x). This solves the non-positive problem we have
    in Expontial class.
    """

    def eval(self, beta, features, y):
        """
        return the pdf at x, with feature feature
        theta = beta * feature.
        fast version.
        :param beta:
        :param feature:
        :param y:
        :return:
        """

        log_miu = np.dot(features, beta)
        tmp = -np.sum(log_miu + y * np.exp(-log_miu))

        if np.isinf(tmp):
            print 'WARNING -- Log likelihood got inf value. It has been replaced by float.max. '
            print 'max of log miu', np.max(log_miu)
            print 'max of  y / miu', np.max(y * np.exp(-log_miu))
            print 'max of y ', max(y)

            return np.finfo(np.float).max
        else:
            return tmp

    def eval_bk(self, beta, features, y):
        """
        return the pdf at x, with feature feature
        theta = beta * feature.
        slow but stable compared with the eval version
        :param beta:
        :param feature:
        :param y:
        :return:
        """
        sum = 0.0

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            sum += inner + y[i] * np.exp(-inner)

        if np.isnan(sum):
            print "get nan sum on calculating Exponential's likelihood."

        return -sum

    def eval_gradient(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        This is a faster version.
        :param beta:
        :param features:
        :param x:
        :return:
        """
        # setup the values we are going to need.
        log_miu = np.dot(features, beta)
        grad_tmp = 1.0 - y * np.exp(-log_miu)

        gradient = -np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient)):
            print 'Warning--The grad_tmp has nan', gradient

        return gradient


    def eval_gradient_bk(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        slow but stable compared with the eval version
        :param beta:
        :param features:
        :param x:
        :return:
        """
        import sys

        gradient = np.zeros(beta.shape[0])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            gradient += features[i] - y[i] * np.exp(-inner) * features[i]

        if np.isnan(gradient[0]) or np.isnan(gradient[1]) or np.isnan(gradient[2]):
            print 'get nan gradient ', gradient
            sys.exit()

        return -gradient

    def eval_hessian(self, beta, features, y):
        """
        return the Hessian matrix of beta at y with feature features.
        dim(beta) = 1. Its length is the dimension of feature space.
        dim(features) = 2. design matrix.
        dim(y) = 1, the observed variables.

        :return: the hessian
        """

        hessian = np.zeros([beta.shape[0], beta.shape[0]])

        for i in range(features.shape[0]):
            inner = np.inner(beta, features[i])
            temp = y[i] * np.exp(-inner)
            for j in range(beta.shape[0]):
                for k in range(beta.shape[0]):
                    hessian[j][k] += temp * features[i][j] * features[i][k]

        return -hessian

    def get_inverse_link(self):
        return np.exp


class  NegativeBinomial(GlmProbDistBase):
    """
    Negative Binomial regression.
    Parameter k is fixed.
    This class is not fully tested. User be aware!
    """
    def __init__(self, k):
        """
        k is the parameter of the negative binomial distribution. This is one single parameter for all the
        observations.
        :param k:
        """
        self.k = k
        self.k_star = np.log(k)

    def eval(self, beta, features, y):
        """
        return the log likelihood, with feature feature
        theta = beta * feature.
        Attention: We omit the ln((y-1)!) in the loglikelihood, because our goal is to optimize the loglikelihood.
        :param beta:
        :param feature:
        :param y:
        :return:
        """
        sum = 0.0

        for i in range(features.shape[0]):
            subsum = 0.0
            miu_i = np.exp(np.inner(features[i], beta))
            for j in range(int(y[i])):
                subsum += np.log(self.k + j)

            sum += subsum + self.k * np.log(self.k) + y[i] * np.log(miu_i) - (self.k + y[i]) * np.log(miu_i + self.k)

        # print '------ Returns the likelihood values as ', sum
        return sum

    def eval_gradient(self, beta, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        This is the gradient against beta. It's not about k, or k_star.
        :param beta:
        :param features:
        :param x:
        :return:
        """

        gradient = np.zeros(beta.shape[0])

        for i in range(features.shape[0]):
            miu_i = np.exp(np.inner(features[i], beta))
            gradient += (self.k * ( y[i] - miu_i ) / (miu_i + self.k)) * features[i]

        return gradient

    def eval_hessian(self, beta, features, y):
        """
        return the Hessian matrix of beta at y with feature features.
        dim(beta) = 1. Its length is the dimension of feature space.
        dim(features) = 2. design matrix.
        dim(y) = 1, the observed variables.

        The hessian is actually not using any value in y.
        :return: the hessian
        """

        hessian = np.zeros([beta.shape[0], beta.shape[0]])

        for i in range(features.shape[0]):
            miu_i = np.exp(np.inner(features[i], beta))
            temp_prod = - self.k * miu_i (( self.k + y[i])/((miu_i + self.k)**2))
            for j in range(beta.shape[0]):
                for k in range(beta.shape[0]):
                    hessian[j][k] += features[i][j] * features[i][k] * temp_prod

        return hessian

    def get_inverse_link(self):
        return np.exp


class  NegativeBinomialWithKstar(GlmProbDistBase):
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
        :param beta:
        :param feature:
        :param y:
        :return:
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

        tmp1 = np.sum(subsum + k * ln_exp_k_star + y * log_miu)
        tmp2 = np.sum((k + y) * (log_miu + log_1_plus_sth))
        tmp = tmp1 - tmp2
        # tmp = np.sum(subsum + k * ln_exp_k_star + y * log_miu - (k + y) * (log_miu + log_1_plus_sth))

        if np.isinf(tmp):
            print 'WARNING -- Log likelihood got inf value. It has been replaced by float.max. '
            print 'max of subsum', np.max(subsum)
            print 'max of y * log miu', np.max(y * log_miu)
            print 'max of (k+y) * log miu and k',  np.max((k + y) * (log_miu + log_1_plus_sth))
            print 'max of log miu and log 1 puls sth', np.max(log_miu), np.max(log_1_plus_sth)
            print 'max of y ', max(y)
            print 'value of  exp and k', k * ln_exp_k_star

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
        :return:
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

        #### gradient of beta
        grad_tmp = (y - miu) / (miu + k)
        ### test of nan in the gradient calculation.
        if np.isnan(np.sum(grad_tmp)):
            if np.isnan(np.sum(miu)):
                print 'The miu has nan', miu
            print 'min of miu + k is ', np.min(miu + k)
            print 'max of miu + k is ', np.max(miu + k)
            print 'min of y - miu is ', np.min(y - miu)
            print 'max of y - miu is ', np.max(y - miu)
            print 'The grad_tmp has nan', grad_tmp

        gradient_beta = k * np.sum(features * grad_tmp.reshape(-1,1), axis=0)
        if np.isnan(np.sum(gradient_beta)):
            print 'The grad_tmp has nan', gradient_beta

        ### derivative of k*
        max_y = int(y.max())
        subsum_y = (1.0 / (np.arange(max_y) + k)).cumsum()
        subsum = subsum_y[y.astype(np.int) - 1]
        subsum[y.astype(np.int) == 0] = 0.0

        derivative_k = np.sum(subsum + 1.0 + beta_k[0] - (k + y)/(k + miu) - (log_miu + log_1_plus_sth))

        if np.isinf(derivative_k):
            print 'WARNING -- Derivative of kstar got inf value. It has been replaced by float.max. '
            derivative_k = np.finfo(np.float).max

        # Assemble them together!
        gradient = np.zeros(beta_k.shape[0])
        gradient[0] = k * derivative_k
        gradient[1:] = gradient_beta

        return gradient

    def eval_backup(self, beta_k, features, y):
        beta = beta_k[1:]
        k = np.exp(beta_k[0])           ## exp(k*).
        ln_exp_k_star = beta_k[0]       ## ln(e^(k*)). It's actually k*, ie. beta_k[0]

        sum = 0.0
        for i in range(features.shape[0]):
            subsum = 0.0
            miu_i = np.exp(np.inner(features[i], beta))
            for j in range(int(y[i])):
                subsum += np.log(k + j)

            if i < 10 or i > features.shape[0] - 11:
                tmp = subsum + k * ln_exp_k_star + y[i] * np.log(miu_i) - (k + y[i]) * np.log(miu_i + k)
                print 'tmp is ', tmp
                print 'miu is ',miu_i
                print 'subsum is ',subsum
            sum += subsum + k * ln_exp_k_star + y[i] * np.log(miu_i) - (k + y[i]) * np.log(miu_i + k)

        return sum

    def eval_gradient_bk(self, beta_k, features, y):
        """
        return the gradient of beta at y with feature features.
        y is the array of the observed values.
        This is the gradient against beta.
        beta_k[0] = k* which is the log of k.
        This is a slower version, compared with the previous version. But it is supposed to be stable.
        :param beta:
        :param features:
        :param x:
        :return:
        """

        beta = beta_k[1:]
        k = np.exp(beta_k[0])           ## exp(k*).

        #### gradient of beta
        gradient_beta = np.zeros(beta.shape[0])

        for i in range(features.shape[0]):
            miu_i = np.exp(np.inner(features[i], beta))
            gradient_beta += (k * ( y[i] - miu_i ) / (miu_i + k)) * features[i]

        ### derivative of k*
        sum = 0.0

        for i in range(features.shape[0]):
            subsum = 0.0
            miu_i = np.exp(np.inner(features[i], beta))
            for j in range(int(y[i])):
                subsum += 1.0 / (k + j)

            sum += subsum + 1.0 + beta_k[0] - (k + y[i])/(k + miu_i) - np.log(miu_i + k)

        #### put them together.
        gradient = np.zeros(beta_k.shape[0])
        gradient[0] = k * sum
        gradient[1:] = gradient_beta

        return gradient

    def eval_hessian(self, beta, features, y):
        """
        return the Hessian matrix of beta at y with feature features.
        :return: the hessian
        """
        raise Exception("Not implemented")

    def get_inverse_link(self):
        return np.exp





