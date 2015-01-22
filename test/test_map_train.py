from __future__ import absolute_import

import unittest
import numpy as np

from mozsci.map_train import TrainModelCV, run_train_models
from mozsci.evaluation import classification_error, auc_wmw_fast
from mozsci.cross_validate import cv_kfold
from mozsci.models import LogisticRegression


class DataTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(5)
        self.X = np.linspace(0, 1, 100).reshape(100, 1)
        self.y = (5 * self.X.reshape(100, ) - 2 + np.random.rand(100) > 0).astype(np.int)

        self.folds = cv_kfold(100, 4, seed=2)

class TestTrainModelCV(DataTest):
    @staticmethod
    def agg_err(yactual, ypred):
        ret = {}
        ret['accuracy'] = classification_error(yactual, ypred)
        ret['auc'] = auc_wmw_fast(yactual, ypred)
        return ret


    def test_map_train_model(self):
        trainer = TrainModelCV([LogisticRegression, classification_error, '/tmp/logistic.json', (), {'lam':0.5}], X=self.X, y=self.y)
        errors = trainer.run()

        # load model
        trained_model = LogisticRegression.load_model('/tmp/logistic.json')
        loaded_model_error = classification_error(self.y, trained_model.predict(self.X))

        # check the errors
        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['train'] - 0.06) < 1e-12)
        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['train'] - loaded_model_error) < 1e-12)

    def test_aggregate_error(self):
        # test an aggregate error function (that returns more than one value)
        trainer = TrainModelCV([LogisticRegression, TestTrainModelCV.agg_err, None, (), {'lam':0.5}],
                       X=self.X, y=self.y, Xtest=self.X[:50, :], ytest=self.y[:50])
        errors = trainer.run()

        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['train']['accuracy'] - 0.06) < 1e-8)
        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['train']['auc'] - 0.99310661764705888) < 1e-8)


    def test_kfold_cv(self):
        trainer = TrainModelCV([LogisticRegression, classification_error, None, (), {'lam':0.5}],
                       X=self.X, y=self.y, folds=self.folds)
        errors = trainer.run()

        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['train'] - 0.063340259665816398) < 1e-12)
        self.assertTrue(np.abs(errors[list(errors.keys())[0]]['test'] - 0.049633305762338022)< 1e-12)


class Test_run_train_models(DataTest):
    def test_run_train_models(self):
        import re

        model_library = [[LogisticRegression, classification_error, None, (), {'lam':0.5}],
          [LogisticRegression, classification_error, None, (), {'lam':50}]]

        errors = run_train_models(2, model_library, X=self.X, y=self.y)
        for k in errors.keys():
            if re.search("{'lam': 0.5}", k):
                err_check = errors[k]

        self.assertTrue(abs(err_check['train'] - 0.06) < 1e-8)


if __name__ == "__main__":
    unittest.main()





