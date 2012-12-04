
import unittest
import numpy as np

from mozsci import pca

class TestLinearPCA(unittest.TestCase):

    def test_linearPCA(self):
        """Test linear PCA"""

        # make the data
        N = 1000
        data = np.zeros((N, 3))
        for k in xrange(N):
            data[k, 0] = (np.random.random() - 0.5) * 5.0 + 2.0
            #data[k, 1] = 3.5 * data[k, 0] + (np.random.random() - 0.5)
            data[k, 1] = (np.random.random() - 0.5) * 5.0
            data[k, 2] = 3.5 + data[k, 0] - 0.55 * data[k, 1] + (np.random.random() - 0.5)

        # make the PCA, do the training, then project on all three eigenvectors,
        # and reconstruct the original data
        p = pca.LinearPCA()
        p.train(data)
        data_proj = p.project(data, 3)

        # reconstruct the data from the projection
        data_reconstruct = np.zeros((N, 3))
        for k in xrange(N):
            data_reconstruct[k, :] = p.mean + data_proj[k, 0] * p.eigvec[:, 0] + data_proj[k, 1] * p.eigvec[:, 1] + data_proj[k, 2] * p.eigvec[:, 2]

        self.assertTrue((np.abs(data_reconstruct - data) < 1.0e-12).all())

        # test out truncate
        self.assertTrue((np.abs(p.truncate(data, 3) - data) < 1.0e-12).all())

        # test json
        json_map = p.to_json()
        p_from_json = pca.LinearPCA(json_map=json_map)
        self.assertEqual(p.nvars, p_from_json.nvars)
        self.assertTrue((np.abs(p.mean - p_from_json.mean) < 1.0e-12).all())
        self.assertTrue((np.abs(p.eigval - p_from_json.eigval) < 1.0e-12).all())
        self.assertTrue((np.abs(p.eigvec - p_from_json.eigvec) < 1.0e-12).all())



if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestLinearPCA)
    suitelist = [suite1]
    suite = unittest.TestSuite(suitelist)
    unittest.TextTestRunner(verbosity=2).run(suite)


