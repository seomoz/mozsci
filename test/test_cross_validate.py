from __future__ import absolute_import

import unittest
import numpy as np
import time

from mozsci import cross_validate


class Test_cv_kfold(unittest.TestCase):
    def test_cv_kfold(self):
        folds = cross_validate.cv_kfold(20, 4, seed=2)

        sum_training = np.sum([len(ele[0]) for ele in folds])
        self.assertTrue(sum_training == 3 * 20)

        sum_training = np.sum([len(ele[1]) for ele in folds])
        self.assertTrue(sum_training == 20)

        actual_folds = [
 [[0, 3, 4, 5, 8, 9, 17, 2, 7, 10, 11, 13, 15, 16, 18], [1, 6, 12, 14, 19]],
 [[1, 6, 12, 14, 19, 2, 7, 10, 11, 13, 15, 16, 18], [0, 3, 4, 5, 8, 9, 17]],
 [[1, 6, 12, 14, 19, 0, 3, 4, 5, 8, 9, 17, 15, 16, 18], [2, 7, 10, 11, 13]],
 [[1, 6, 12, 14, 19, 0, 3, 4, 5, 8, 9, 17, 2, 7, 10, 11, 13], [15, 16, 18]]]

        self.assertEqual(actual_folds, folds)


if __name__ == "__main__":
    unittest.main()






