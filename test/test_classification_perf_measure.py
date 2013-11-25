import unittest
import numpy as np

from mozsci import classification_perf_measure


class TestClassificationPerfMeasure(unittest.TestCase):

    def test_basic_measure_1(self):
        """
        Test classification_model_performance. All correct case.
        """
        observed = np.array([0, 1, 1, 0, 0, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance(observed, calculated)

        self.assertEqual(measure, 0)

    def test_basic_measure_2(self):
        """
        Test classification_model_performance. All correct case.
        """
        observed = np.array([0, 1, 0, 1, 0, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance(observed, calculated)

        self.assertAlmostEqual(measure, 0.2857142857140)

    def test_basic_measure_3(self):
        """
        Test classification_model_performance. weighted case.
        """
        observed = np.array([0, 1, 0, 1, 0, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance(observed, calculated, [1.0, 3.0])

    def test_matrix_measure_1(self):
        """
        Test classification_model_performance_matrix. All correct case.
        """
        observed = np.array([0, 1, 1, 0, 0, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance_matrix(observed, calculated)
        expected_measure = np.array([[4, 0], [0, 3]])

        np.testing.assert_array_almost_equal(measure, expected_measure)

    def test_matrix_measure_2(self):
        """
        Test classification_model_performance_matrix. All correct case.
        """
        observed = np.array([0, 1, 0, 1, 0, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance_matrix(observed, calculated)
        expected_measure = np.array([[3, 1], [1, 2]])

        np.testing.assert_array_almost_equal(measure, expected_measure)

    def test_matrix_measure_3(self):
        """
        Test classification_model_performance_matrix. multiple classes case.
        """
        observed = np.array([1, 0, 1, 0, 1, 0, 2, 3])
        calculated = np.array([1, 0, 1, 1, 0, 2, 3, 0])

        measure = classification_perf_measure.classification_model_performance_matrix(observed, calculated)
        expected_measure = np.array([[1, 1, 1, 0], [1, 2, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]])

        np.testing.assert_array_almost_equal(measure, expected_measure)

    def test_loss_measure_1(self):
        """
        Test classification_model_performance_loss. default loss (0-1 loss).
        """
        observed = np.array([0, 1, 1, 0, 1, 0, 1])
        calculated = np.array([0, 1, 1, 0, 0, 0, 1])

        measure = classification_perf_measure.classification_model_performance_loss(observed, calculated)

        self.assertEqual(measure, 1)

    def test_loss_measure_2(self):
        """
        Test classification_model_performance_loss. user defined loss measure - squared loss.
        """
        observed = np.array([0, 1, 0, 1, 0, 2, 1])
        calculated = np.array([0, 1, 1, 0, 2, 0, 1])

        loss = lambda i, j: (i-j)*(i-j)

        measure = classification_perf_measure.classification_model_performance_loss(observed, calculated, loss)

        self.assertEqual(measure, 10)


if __name__ == "__main__":
    unittest.main()