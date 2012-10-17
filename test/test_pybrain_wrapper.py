
import unittest
from mozsci.models.pybrain_wrapper import PyBrainNN
import numpy as np

class Testpybrain_wrapper(unittest.TestCase):

    def test_xor(self):
        """Test learning XOR with a neural net"""
        X = np.array([[0.0, 0.0],
               [0, 1],
               [1, 0],
               [1, 1]])
        y = np.array([0, 1, 1, 0])
        net = PyBrainNN(learning_rate=0.1, maxiterations=10000, lam=0.0, args=(2, 3, 1), kwargs={'fast':True, 'bias':True})
        net.fit(X, y)

        ypred = net.predict(X)
        ypred_int = (ypred > 0.5).astype(np.int)
        self.assertTrue(np.allclose(ypred_int, y))


if __name__ == "__main__":
    unittest.main()





