
import unittest
from mozsci.models.pybrain_wrapper import PyBrainNN
import numpy as np
import json

class Testpybrain_wrapper(unittest.TestCase):

    def train_xor(self, learning_rate=0.1):
        """Trains XOR with a neural net"""
        X = np.array([[0.0, 0.0],
               [0, 1],
               [1, 0],
               [1, 1]])
        y = np.array([0, 1, 1, 0])
        net = PyBrainNN(learning_rate=learning_rate, maxiterations=10000, lam=0.0, args=(2, 3, 1), kwargs={'bias':True})
        net.fit(X, y)

        return net, X, y


    def test_xor(self):
        net, X, y = self.train_xor()

        ypred = net.predict(X)
        ypred_int = (ypred > 0.5).astype(np.int)

        self.assertTrue(np.allclose(ypred_int, y))

    def test_load_save_model(self):
        import tempfile

        net, X, y = self.train_xor()

        # save the model
        tfile = tempfile.TemporaryFile()
        net.save_model(tfile)

        # load in the model
        tfile.seek(0)

        model = json.load(tfile)

        netloaded = PyBrainNN.load_model(model)

        self.assertTrue(np.allclose(net.predict(X), netloaded.predict(X)))

    def test_learning_rate(self):
        net, X, y = self.train_xor(5.0)
        ypred = net.predict(X)
        ypred_int = (ypred > 0.5).astype(np.int)
        self.assertTrue(np.allclose(ypred_int, y))
        self.assertTrue((np.abs(y - ypred) < 1e-3).all())



if __name__ == "__main__":
    unittest.main()



