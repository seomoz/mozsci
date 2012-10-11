
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




class Test_spearmanr_by(unittest.TestCase):

    def test_spearmanr_by(self):

        f = np.array([50,  52.19589972,  44.97281905,  50,
            47.6719409 ,  45.96619825,  50,  50,
            48.18824048,  54.88529706,  42.67667074,  41.80373588,
            37.29934119,  57.98812747,  45.04782628,  38.10858417,
            46.44031713,  40.59823939,  26.29936944,  23.96820474,
            47.98343799,  36.4455311 ,  43.92931621,  55.19172514,
            33.44633285,  37.38381116,  39.03392758,  41.43285553,
            28.63082987,  31.86069758,  41.19551474,  29.04928565,
            39.09690404,  36.75441683,  29.66390582,  70.4035713 ,
            63.53532854,  49.78916058,  64.39911984,  65.41353192,
            48.42353021,  60.38572122,  42.44357922,  42.86378695,
            58.93821467,  61.93862217,  36.23459784,  64.57533596,
            40.09399141,  45.57233379,  44.7748158 ,  50.88705955,
            47.24016865,  51.75866967,  36.17935042,  46.73933887,
            52.7136634 ,  47.0337377 ,  34.19077012,  18.5836512 ,
            41.63257011,   9.8698871 ,  37.63277795,  47.71676464,
            34.89667886,  35.10845963,  44.56638481,  36.70884056,
            57.9185177 ,  50.65260932,  58.53307806,  43.25154747,
            40.59802125,  38.97005406,  35.19682907,  51.94755877,
            44.04430199,  35.84048228,  36.25006727,  46.35317423,
            37.44668618,  16.90596421,  38.87970562,  47.33515849,
            27.41230181,  29.47142008])
    
        position = np.array([1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10.,  12.,
            13.,  15.,  16.,  17.,  19.,  23.,  24.,  25.,  26.,  27.,  28.,
            29.,   1.,   2.,   3.,   6.,   8.,   9.,  11.,  12.,  13.,  17.,
            19.,  21.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,
            10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,
            22.,  23.,  24.,  25.,  26.,  27.,   1.,   2.,   4.,   5.,   6.,
             7.,   8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.,  16.,  17.,
            18.,  20.,  21.,  22.,  23.,  24.,  25.,  26.,  27.])
    
        queryid = np.array([0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  2,  2,  2,  2,
            2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
            2,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,
            3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
            3,  3,  3,  3,  3,  3,  3,  3], np.int)

        fast_spearman = evaluation.spearmanr_by(f, position, queryid)
        self.assertTrue(abs(fast_spearman - -0.42666971560358913) < 1e-8)


if __name__ == "__main__":
    unittest.main()



