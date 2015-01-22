from __future__ import absolute_import

import unittest
import numpy as np
from mozsci import inputs
from six.moves import range

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


class TestLogScaledTransformer(unittest.TestCase):
    def test_log_transformer(self):
        mean = np.array([0.5, 1.0])
        std = np.array([0.3, 0.8])
        offset = 2.0
        nsamples = int(1e6)
        samples = np.zeros((nsamples, 2))
        for k in range(2):
            samples[:, k] = np.random.normal(mean[k], std[k], nsamples)
        exp_samples = np.exp(samples) - offset

        transformer = inputs.LogScaledTransformer(offset=offset)

        # check fit
        transformer.fit(exp_samples)
        self.assertTrue(np.allclose(transformer.mean_, mean, atol=1e-2))
        self.assertTrue(np.allclose(transformer.std_, std, atol=1e-2))

        # check transform
        X = exp_samples[:10]
        transformed = transformer.transform(X)
        expected = 1.0 / transformer.std_ * (
            np.log(X + offset) - transformer.mean_)
        self.assertTrue(np.allclose(transformed, expected))

        # inverse transform
        self.assertTrue(np.allclose(X,
            transformer.inverse_transform(transformer.transform(X))))

class TestBucketTransformer(unittest.TestCase):
    def test_bucket_transformer(self):
        transformer = inputs.BucketTransformer([0, 1, 2.4])
        X = np.array([0.5, 1.2, -1, 3.9, 1.9, 2.1])
        Y = transformer.transform(X)
        expectedY = np.array(
                [[ 0.,  1.,  0.,  0.],
                [ 0.,  0.,  1.,  0.],
                [ 1.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  1.],
                [ 0.,  0.,  1.,  0.],
                [ 0.,  0.,  1.,  0.]]
        )
        self.assertTrue(np.allclose(Y, expectedY))


if __name__ == "__main__":
    unittest.main()

