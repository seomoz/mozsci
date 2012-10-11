
import unittest
import numpy as np

from mozsci import evaluation
from mozsci.inputs import mean_std_weighted


class TestAUCFast(unittest.TestCase):
    def test_auc_wmw_fast(self):

        t = [-1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, -1, 1]
        p = [0.01, 0.05, 0.2, 0.25, 0.1, 0.9, 0.6, 0.01, 0.90, 1.0, 0.33, 0.55, 0.555]

        auc_act = 0.54761904761904767
        auc = evaluation.auc_wmw_fast(t, p)

        self.assertTrue(abs(auc_act - auc) < 1.0e-8)


class Testclassification_error(unittest.TestCase):
    def test_classification_error(self):
        y = np.array([0, 1, 1, 0])
        ypred = np.array([0.1, 0.9, 0.4, 0.2])

        self.assertTrue(abs(evaluation.classification_error(y, ypred) - 0.25) <
 1e-12)
        self.assertTrue(abs(evaluation.classification_error(y, ypred, thres=0.3
) - 0.0) < 1e-12)

        weights = np.array([1.0, 0.8, 0.7, 0.6])
        self.assertTrue(abs(evaluation.classification_error(y, ypred, weights=weights) - (1.0 - (1.0 + 0.8 + 0.6) / (weights.sum()))) < 1.0e-12)




class Test_precision_recall_f1(unittest.TestCase):

    def setUp(self):
        self.yactual = np.array([0, 0, 0, 0, 1, 1, 1])
        self.ypred =   np.array([0, 1, 1, 1, 1, 0, 0])
        self.weights = np.array([1, 2, 3, 4, 5, 6, 7])

        self.yactual1 = self.yactual.reshape(7, 1)
        self.ypred1 = self.ypred.reshape(1, 7)
        self.weights1 = self.weights.reshape(1, 7)

    def test_precision_recall_f1(self):
        tp = 1.0
        fp = 3.0
        fn = 2.0

        actual_prec_rec_f1 = Test_precision_recall_f1.prec_rec_f1_from_tp_fp_fn(tp, fp, fn)
        for y in [self.yactual, self.yactual1]:
            for ypred in [self.ypred, self.ypred1]:
                prec_rec_f1 = evaluation.precision_recall_f1(y, ypred)
                for k in xrange(3):
                    self.assertTrue(abs(actual_prec_rec_f1[k] - prec_rec_f1[k]) < 1e-12)

    def test_precision_recall_f1_weighted(self):
        tp = 5.0
        fp = 2.0 + 3 + 4
        fn = 6.0 + 7

        actual_prec_rec_f1 = Test_precision_recall_f1.prec_rec_f1_from_tp_fp_fn(tp, fp, fn)

        for y in [self.yactual, self.yactual1]:
            for ypred in [self.ypred, self.ypred1]:
                for weights in [self.weights, self.weights1]:
                    prec_rec_f1 = evaluation.precision_recall_f1(y, ypred, weights=weights)
                    for k in xrange(3):
                        self.assertTrue(abs(actual_prec_rec_f1[k] - prec_rec_f1[k]) < 1e-12)


    @staticmethod
    def prec_rec_f1_from_tp_fp_fn(tp, fp, fn):
        actual_prec_rec_f1 = np.zeros(3)
        actual_prec_rec_f1[0] = tp / (tp + fp) # precision
        actual_prec_rec_f1[1] = tp / (tp + fn) # recall
        actual_prec_rec_f1[2] = 2.0 * actual_prec_rec_f1[0] * actual_prec_rec_f1[1] / (actual_prec_rec_f1[0] + actual_prec_rec_f1[1])  # f1
        return actual_prec_rec_f1



class Test_pearson_weighted(unittest.TestCase):
    def test_pearson_weighted(self):
        from scipy.stats import pearsonr

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([1.0, 1.5, -0.5, 3.4, 2.9])
        weights = np.array([1, 0, 0.5, 2, 1.5])

        r_no_wgt = pearsonr(x, y)[0]
        r_no_wgt_test = evaluation.pearsonr_weighted(x, y)
        r_ones_wgt = evaluation.pearsonr_weighted(x, y, np.ones(x.shape))

        self.assertTrue(abs(r_no_wgt - r_no_wgt_test) < 1e-12)
        self.assertTrue(abs(r_no_wgt - r_ones_wgt) < 1e-12)

        xm = mean_std_weighted(x, weights)
        ym = mean_std_weighted(y, weights)
        r_wgt = np.sum((x - xm['mean']) * (y - ym['mean']) * weights) / np.sum(weights)
        self.assertTrue((evaluation.pearsonr_weighted(x, y, weights) - r_wgt) < 1e-12)


if __name__ == "__main__":
    unittest.main()



