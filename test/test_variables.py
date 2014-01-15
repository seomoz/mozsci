
import unittest
import numpy as np

from mozsci.evaluation import classification_error
from mozsci.inputs import IdentityTransformer, LogScaledTransformer
from mozsci import variables
from sklearn.linear_model import LogisticRegression


class TestModelDriver(unittest.TestCase):
    def test_model_driver(self):
        independents = [
            variables.Variable('x0', IdentityTransformer()),
            variables.Variable('x1', LogScaledTransformer())
        ]
        dependents = [variables.Variable('y', IdentityTransformer())]
        model_variables = variables.ModelVariables(independents, dependents)

        # make some test data
        N = int(1e5)
        data = np.zeros(
            N, dtype=[('x0', np.float), ('x1', np.float), ('y', np.int)])
        np.random.seed(5)
        data['x0'] = np.random.rand(N)
        data['x1'] = np.random.normal(0.5, 2.0, N)
        data['y'] = 3 * data['x0'] - 2 * data['x1'] - 1.5 > 0.0

        # rescale x1 
        data['x1'] = np.exp(data['x1'])

        # create driver and fit
        model = variables.ModelDriver(model_variables, LogisticRegression(C=1e5))

        # first try to fit with regular numpy arrays
        X = data.view(dtype=np.float).reshape(-1, 3)[:, :2]
        y = data.view(dtype=np.int).reshape(-1, 3)[:, 2].reshape(-1, 1)
        model.fit(X, y)
        ypred = model.predict(X)
        self.assertTrue(classification_error(y, ypred) < 0.002)

        # now try using __getitem__
        model.fit(data, data)
        ypred = model.predict(data)
        self.assertTrue(classification_error(data['y'], ypred) < 0.002)

        # serialization
        model_string = model.dumps()
        model_loaded = variables.ModelDriver.loads(model_string)
        self.assertTrue(np.allclose(
            model.predict(data, predict_prob=True),
            model_loaded.predict(data, predict_prob=True)))


if __name__ == "__main__":
    unittest.main()
