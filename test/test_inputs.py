
import unittest
import numpy as np
from mozsci import inputs

class Test_mean_std_weightd(unittest.TestCase):
    def test_mean_std(self):

        # test 1D case
        x = np.array([1, 2, 3, 4, 5])
        weights = np.array([0.2, 0.1, 2,0.5, 1])

        ret = inputs.mean_std_weighted(x)
        self.assertTrue(abs(ret['mean'] - 3.0) < 1e-8)
        self.assertTrue(abs(ret['std'] - np.sqrt(2 * (4 + 1) / 5)) < 1e-8)

        ret = inputs.mean_std_weighted(x, np.ones(x.shape))
        self.assertTrue(abs(ret['mean'] - 3.0) < 1e-8) 
        self.assertTrue(abs(ret['std'] - np.sqrt(2 * (4 + 1) / 5)) < 1e-8)

        ret = inputs.mean_std_weighted(x, weights)
        m = np.sum(weights * x) / np.sum(weights)
        s = np.sqrt(np.sum((x - m)**2 * weights) / np.sum(weights))
        self.assertTrue(abs(ret['mean'] - m) < 1e-8)
        self.assertTrue(abs(ret['std'] - s) < 1e-8)

        # 2D case
        x = np.array([[1, 2],
                     [-0.5, 0.0],
                     [3, -0.55]])
        weights = np.array([0.5, 2, 1.55])

        ret = inputs.mean_std_weighted(x, weights)

        sum_weights = np.sum(weights)
        m1 = (1.0 * 0.5 + -0.5 * 2 + 3 * 1.55) / sum_weights
        m2 = (2.0 * 0.5 + 0.0 * 2 + -0.55 * 1.55) / sum_weights
        self.assertTrue(np.allclose(ret['mean'], [m1, m2]))

        s1 = np.sqrt(((1.0 - m1) ** 2 * 0.5 + (-0.5 - m1)**2 * 2.0 + (3 - m1)**2 * 1.55) / sum_weights)
        s2 = np.sqrt(((2 - m2) ** 2 * 0.5 + (0.0 - m2)**2 * 2.0 + (-0.55 - m2)**2 * 1.55) / sum_weights)
        self.assertTrue(np.allclose(ret['std'], [s1, s2]))


if __name__ == "__main__":
    unittest.main()

