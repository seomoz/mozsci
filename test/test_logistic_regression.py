from __future__ import absolute_import

import unittest
from mozsci.models import LogisticRegression
import numpy as np

class TestLogisticRegression(unittest.TestCase):

    def setUp(self):
        self.x = np.array([[1, -2], [-0.5, -2]])
        self.t = np.array([0, 1])
        self.w = np.array([3, -1])
        self.b = 1
        self.lam = 7

    def test_sigmoid(self):
        y = LogisticRegression._sigmoid(self.x, self.b, self.w)
        yact = np.array([1.0 / (1.0 + np.exp(-6)), 1.0 / (1.0 + np.exp(-1.5))])

        self.assertTrue(np.all(np.abs(y - yact) < 1.0e-12))

    def test_error_gradient(self):
        x0 = np.array([self.x[0]])
        x1 =  np.array([self.x[1]])
        error, gradient = LogisticRegression._loss_gradient(x0, x1, self.b, self.w, self.lam)

        # this assumes test_sigmoid pases
        err_act = -np.log(LogisticRegression._sigmoid(x1, self.b, self.w)) - np.log(1.0 - LogisticRegression._sigmoid(x0, self.b, self.w)) + 0.5 * 7 * 10
            
        pred_error = LogisticRegression._sigmoid(self.x, self.b, self.w) - self.t
        gradient_act = np.array([0.0, 7 * 3, 7 * -1])
        gradient_act[0] = np.sum(pred_error)
        gradient_act[1] += np.sum(pred_error * self.x[:, 0])
        gradient_act[2] += np.sum(pred_error * self.x[:, 1])

        self.assertTrue( abs(float(err_act) - error) < 1.0e-12 ) 
        self.assertTrue(np.all(np.abs(gradient - gradient_act) < 1.0e-12))

        # weighted case
        x00 = np.array([self.x[0], [55, -2]])
        error_weighted, gradient_weighted = LogisticRegression._loss_gradient(x00, x1, self.b, self.w, self.lam, [np.array([0.4, 0.75]), np.array(0.35)])
        err_weighted_act = -np.log(LogisticRegression._sigmoid(x1, self.b, self.w)) * 0.35 - np.log(1.0 - LogisticRegression._sigmoid(x0, self.b, self.w)) * 0.4 - np.log(1.0 - LogisticRegression._sigmoid([x00[1, :]], self.b, self.w)) * 0.75 + 0.5 * 7 * 10
        self.assertTrue( abs(float(err_weighted_act) - error_weighted) < 1.0e-12 )

    def test_fit(self):
        from mozsci.evaluation import classification_error
        np.random.seed(5)
        N = int(1e5)
        x = np.random.rand(N, 2)
        y = (3 * x[:, 0] - 2 * x[:, 1] - 1.5 > 0.0).astype(np.int)
        lr = LogisticRegression()
        lr.fit(x, y, factr=1e4)
        ypred = lr.predict(x)
        self.assertTrue(classification_error(y, ypred) < 0.002)



if __name__ == "__main__":
    unittest.main()


