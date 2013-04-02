import unittest
import numpy as np

from mozsci.glm import prob_distributions
from mozsci.glm import regularization
from mozsci.glm import simplified_glm

class TestGlm(unittest.TestCase):

    def test_negative_binomial_dist_likelihood(self):
        """
        Test the calculation of the log likelihood of the negative binomial distribution.
        :return:
        """
        features = np.array([
            [1,56.98883,42.45086, 1.0],
            [1,37.09416,46.82059, 1.0],
            [0,32.27546,43.56657, 1.0],
            [0,29.05672,43.56657, 1.0],
            [0,6.748048,27.24847, 1.0],
            [0,61.65428,48.41482, 1.0]
        ])

        Y = np.array([4,  4,  2,  3,  3,  13 ])
        beta_k = np.array([10.0,  0,  0,  0,  0])

        dist = prob_distributions.NegativeBinomialWithKstar()

        calculated = dist.eval(beta_k, features, Y)

        self.assertAlmostEqual(calculated, -5.9967772892)


    def test_negative_binomial_dist_gradient(self):
        """
        Test the gradient of the log likelihood of negative binomial distribution.
        """
        # input data.
        features = np.array([
            [1,56.98883,42.45086, 1.0],
            [1,37.09416,46.82059, 1.0],
            [0,32.27546,43.56657, 1.0],
            [0,29.05672,43.56657, 1.0],
            [0,6.748048,27.24847, 1.0],
            [0,61.65428,48.41482, 1.0]
        ])
        Y = np.array([4,  4,  2,  3,  3,  13 ])
        beta_k = np.array([10.0,  0,  0,  0,  0])

        # expected output
        expected = np.array([-3.22202699e-03, 5.99972761e+00, 1.12593421e+03, 1.03394190e+03, 2.29989558e+01])

        # calculation.
        dist = prob_distributions.NegativeBinomialWithKstar()

        calculated = dist.eval_gradient(beta_k, features, Y)

        np.testing.assert_almost_equal(calculated, expected, decimal=5)

    def test_poisson_regression(self):
        """
        This method is used to test the poisson regression works as it should.
        The data is from: http://www.oxfordjournals.org/our_journals/tropej/online/ma_chap13.pdf
        :return:
        """
        features = np.array( [
            [236,0], [739,1], [970,1], [2371,1], [309,1], [679,1], [26,0], [1272,1], [3246,1], [1904,1],
            [357,1], [1080,1], [1027,1], [28,0], [2507,1], [138,0], [502,1], [1501,1], [2750,1], [192,1], ] )

        Y = np.array([ 8, 16, 15, 23, 5, 13, 4, 19, 33, 19, 10, 16, 22, 2, 22, 2, 18, 21, 24, 9])

        regular = regularization.NullRegularization()

        # or we can use regular = regularization.NullRegularization()
        reg = simplified_glm.PoissonRegression(lam=0)
        reg.fit(features, Y)

        # The correct result should be [0.00033, 1.045, 1.351], The last one is the constant.
        # bfgs gives [  3.26073590e-04   1.04513753e+00   1.35099878e+00]
        expected = np.array([0.00033, 1.045, 1.351])
        np.testing.assert_almost_equal(reg.params, expected, decimal=2)

    def test_negative_binomial(self):
        """
        This method is used to test the negative binomial 'regression' works as it should.
        Data is from : http://www.ats.ucla.edu/stat/sas/dae/negbinreg.htm
        What they got: loglikelihood: Log Likelihood                         2149.3649
            Parameter     DF    Estimate       Error           Limits           Chi-Square    Pr > ChiSq
            Intercept      1      2.7161      0.2326      2.2602      3.1719        136.38        <.0001
            male           1     -0.4312      0.1397     -0.7049     -0.1574          9.53        0.0020
            math           1     -0.0016      0.0048     -0.0111      0.0079          0.11        0.7413
            langarts       1     -0.0143      0.0056     -0.0253     -0.0034          6.61        0.0102
            Dispersion     1      1.2884      0.1231      1.0471      1.5296

            NOTE: The negative binomial dispersion parameter was estimated by maximum likelihood.

        What we got: (Under the same condition - no regularization. No max iteration limit.)
            the likelihood term value and the regularization term value are  -2149.36485714 0.0
            Optimization terminated successfully.
            Current function value: -2149.364857
            Iterations: 27
            Function evaluations: 184
            Gradient evaluations: 161
            The linear coefficients are:
            [ (This is k*) -2.53387660e-01  -4.31184391e-01  -1.60095828e-03  -1.43475268e-02
              (This is the intercept) 2.71606920e+00]
        """
        mydata = np.genfromtxt('data/poissonreg.csv', delimiter=',', skip_header=1)
        features = mydata[:, 2:5]

        Y = mydata[:, 6]

        reg = simplified_glm.NegativeBinomialWithKstarRegression(3 + 2, lam=0)
        reg.fit(features, Y)

        ## data from ucla.
        expected = np.array([-0.4312, -0.0016, -0.0143, 2.7161])

        np.testing.assert_almost_equal(reg.params[1:], expected, decimal=2)

if __name__ == "__main__":
    unittest.main()
