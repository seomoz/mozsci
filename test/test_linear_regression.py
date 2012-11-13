
import unittest
from mozsci.models import LinearRegression
from mozsci.evaluation import pearsonr_weighted
import numpy as np

class TestLogisticRegression(unittest.TestCase):
    def test_linear_regression(self):
        np.random.seed(55)
        X = np.random.rand(1000, 3)
        w = [0.5, 1.3, -2.5]
        b = 12.5
        y = X[:, 0] * w[0] + X[:, 1] * w[1] + X[:, 2] * w[2] + b

        # should convert to the exact solution with only a little regularization
        lr = LinearRegression(lam=0.001)
        lr.fit(X, y)
        ypred = lr.predict(X)
        self.assertTrue(pearsonr_weighted(y, ypred) > 0.99)

        # try weighted
        weights = np.random.rand(1000)
        lr = LinearRegression(lam=0.001)
        lr.fit(X, y, weights=weights)
        ypred = lr.predict(X)
        self.assertTrue(pearsonr_weighted(y, ypred, weights))


if __name__ == "__main__":
    unittest.main()


