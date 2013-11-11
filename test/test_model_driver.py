"""
This python file contains all the test cases for parked domain model module.
"""

import unittest
import numpy as np

# from baroquery import parked_domain_model
from numpy.testing import assert_array_almost_equal, assert_almost_equal
from sklearn import neighbors, datasets, linear_model, svm
from mozsci import model_driver


class TestDumpsVariableDef(unittest.TestCase):
    """
    Test cases for dumps_variable_def.
    """
    # some data for the following test cases.
    log_bing_volume = {'name': 'bing_log_volume', 'transform': lambda x: x + 1, 'normalization': True}
    log_fwe_volume = {'name': 'fwe_log_volume', 'transform': lambda x: x + 1, 'normalization': True}
    variable_definition_for_test = {'independent_variables': [log_bing_volume, log_fwe_volume],
                                    'dependent_variable': {'name': 'google_volume',
                                                           'transform': lambda x: x + 1, 'normalization': False},
                                    'data_schema': [log_bing_volume, log_fwe_volume]}

    def test_dumps_json(self):
        """
        Test the json string dumped from the variable definition.
        """
        dumped = model_driver.dumps_variable_def(TestDumpsVariableDef.variable_definition_for_test)
        expected = {'independent_variables': [{'name': 'bing_log_volume', 'normalization': True},
                                              {'name': 'fwe_log_volume', 'normalization': True}],
                    'dependent_variable': {'name': 'google_volume', 'normalization': False},
                    'data_schema': ['bing_log_volume', 'fwe_log_volume']}

        self.assertEqual(dumped, expected)


class TestModelDriver(unittest.TestCase):
    """
    Test cases for ModelDriver.

    I think there are only three things worth to test: fit(), predict() and load. The normalization can be
    tested in the prediction and/or fit functions.

    for those cases we can use a simple logistic regression model.
    """
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target

    iris_1st_feature = {'name': 'iris_1st_feature', 'transform': lambda x: x, 'normalization': False}
    iris_2nd_feature = {'name': 'iris_2nd_feature', 'transform': lambda x: x, 'normalization': False}
    variable_definition_for_test = {'independent_variables': [iris_1st_feature, iris_2nd_feature],
                                    'dependent_variable': {'name': 'iris_target',
                                                           'transform': lambda x: x, 'normalization': False},
                                    'data_schema': [iris_1st_feature, iris_2nd_feature]}

    def test_fit_model_lr_1(self):
        """
        This is a test case without normalization.
        """
        model = linear_model.LogisticRegression(C=1e5)
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1,
                    1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1,
                    1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1,
                    2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1]

        self.assertListEqual(list(predicted), expected)
        # print 'un-normalized model:', driver.serialize_model()

    def test_fit_model_lr_2(self):
        """
        This is a test case without normalization.
        """
        normalized_iris_1st_feature = {'name': 'iris_1st_feature', 'transform': lambda x: x, 'normalization': True}
        normalized_iris_2nd_feature = {'name': 'iris_2nd_feature', 'transform': lambda x: x, 'normalization': True}
        variable_definition_local_for_test = {'independent_variables': [normalized_iris_1st_feature,
                                                                  normalized_iris_2nd_feature],
                                        'dependent_variable': {'name': 'iris_target',
                                                               'transform': lambda x: x, 'normalization': False},
                                        'data_schema': [normalized_iris_1st_feature, normalized_iris_2nd_feature]}

        model = linear_model.LogisticRegression(C=1e5)
        driver = model_driver.ModelDriver(variable_definition_local_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1,
                    1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 1,
                    1, 1, 1, 1, 2, 1, 1, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 1, 2, 1, 2, 1,
                    2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 1]

        self.assertListEqual(list(predicted), expected)
        # print 'normalized model:', driver.serialize_model()

    def test_fit_model_svm_1(self):
        """
        This is a test case without normalization.
        """
        model = svm.LinearSVC(C=1e5, loss='l1')
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 1, 2, 2, 2, 1, 2, 0, 1, 2,
                    1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 0, 2, 2, 1, 2, 1, 1, 2, 1,
                    1, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        self.assertListEqual(list(predicted), expected)
        # print 'un-normalized svm model:', driver.serialize_model()

    def test_fit_model_random_forest_1(self):
        """
        This is a test case without normalization.
        """
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=30)
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        # print 'predicted is ', predicted

        expected = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  2.,  1., 2.,  1.,  2.,
                    1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  1.,  1.,  1.,
                    1.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1., 1.,  2.,  1.,  1.,  1.,
                    1.,  1.,  1.,  1.,  1.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  1.,
                    2.,  2.,  2.,  2.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  2.,  2.,
                    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  2., 2.,  2.,  2.,  2.,  2.,  1.]

        error_cnt = sum(expected[ii] != predicted[ii] for ii in xrange(len(predicted)))
        self.assertLess(error_cnt, 20)

    def test_fit_model_extrememly_randomized_trees_1(self):
        """
        This is a test case without normalization.

        """
        from sklearn.ensemble import ExtraTreesClassifier

        model = ExtraTreesClassifier(n_estimators=30, max_depth=None, min_samples_split=1, random_state=0)
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        expected = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  2.,  1., 2.,  1.,  2.,
                    1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  1.,  1.,  1.,
                    1.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1., 1.,  2.,  1.,  1.,  1.,
                    1.,  1.,  1.,  1.,  1.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  1.,
                    2.,  2.,  2.,  2.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  2.,  2.,
                    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  2., 2.,  2.,  2.,  2.,  2.,  1.]

        error_cnt = sum(expected[ii] != predicted[ii] for ii in xrange(len(predicted)))
        self.assertLess(error_cnt, 20)

    def test_fit_model_gradient_boosting_classifier_1(self):
        """
        This is a test case without normalization.

        """
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(n_estimators=30, max_depth=1, random_state=0)
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        expected = [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,
                    0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,  2.,  1., 2.,  1.,  2.,
                    1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1.,  1.,  1.,  1., 2.,  1.,  1.,  1.,
                    1.,  2.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  2.,  1.,  1.,  1., 1.,  2.,  1.,  1.,  1.,
                    1.,  1.,  1.,  1.,  1.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  1.,
                    2.,  2.,  2.,  2.,  2.,  1.,  2.,  2.,  2.,  2.,  2.,  2., 2.,  2.,  2.,  2.,  2.,  2.,  2.,
                    2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  2.,  1.,  2., 2.,  2.,  2.,  2.,  2.,  1.]

        error_cnt = sum(expected[ii] != predicted[ii] for ii in xrange(len(predicted)))
        self.assertLess(error_cnt, 20)

    def test_fit_model_pybrain_neural_network_1(self):
        """
        This is a test case without normalization.
        """
        def separate(number):
            if number <= 0.5:
                return 0
            elif number <= 1.5:
                return 1
            else:
                return 2
        from mozsci.models.pybrain_wrapper import PyBrainNN, PyBrainNNError

        model = PyBrainNN(learning_rate=0.1, maxiterations=200, lam=0.0, args=(2, 3, 1),
                          kwargs={'fast': True, 'bias': True})
        driver = model_driver.ModelDriver(TestModelDriver.variable_definition_for_test, model=model)

        driver.fit(TestModelDriver.X, TestModelDriver.Y)

        predicted = driver.predict(TestModelDriver.X, make_copy=True)

        predicted_integers = [separate(item) for item in predicted]

        # The following is the observed values.
        expected = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

        error_cnt = sum(expected[ii] != predicted_integers[ii] for ii in xrange(len(predicted)))
        self.assertLess(error_cnt, 120)


if __name__ == "__main__":
    unittest.main()