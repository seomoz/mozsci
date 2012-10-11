"""Evaluate model performance including efficient C implementations"""


import numpy as np
import scipy.weave

from .inputs import mean_std_weighted

def pearsonr_weighted(x, y, weights=None):
    """Weighted Pearson correlation coefficient.

    x, y = (N, ) numpy arrays or 
    weights = (N, ) or None for no weights"""
    from scipy.stats import pearsonr
    if weights is None:
        return pearsonr(x, y)[0]
    else:
        mean_std_x = mean_std_weighted(x.flatten(), weights.flatten())
        mean_std_y = mean_std_weighted(y.flatten(), weights.flatten())
        cov_xy = np.sum((x - mean_std_x['mean']) * (y - mean_std_y['mean']) * weights.flatten()) / np.sum(weights)
        return cov_xy / mean_std_x['std'] / mean_std_y['std']   # r


def auc_wmw_fast(t, p, weights=None):
    """Compute the AUC by using the Wilcoxon-Mann-Whitney
    statistic. Only binary classification problems are allowed.

    :Parameters:
      t : 1d array_like object integer (-1/+1) or (0/1)
        target values
      p : 1d array_like object (negative/positive values)
        predicted values
      
    :Returns:
      AUC : float, in range [0.0, 1.0]

    A fast 

    weights = a length(t) array with the weights
        if omitted, uses uniform weights
    """
    tarr = np.asarray(t, dtype=np.int)
    parr = np.asarray(p, dtype=np.float)

    if len(tarr) != len(parr):
        raise ValueError("t, p: shape mismatch")

    idxp = np.where(tarr ==  1)[0]
    idxn = np.where(tarr <= 0)[0]
    nidxn = idxn.shape[0]
    nidxp = idxp.shape[0]

    if weights is not None:
        warr = np.asarray(weights, dtype=np.float)
    else:
        warr = np.ones(tarr.shape)

    code = """
        double auc = 0.0;
        double sum_weights = 0.0;
        for (int i=0; i < nidxp; i++)
        {
            for (int j=0; j < nidxn; j++)
            {
                double this_weight = warr(idxp(i)) + warr(idxn(j));
                sum_weights += this_weight;
                if (parr(idxp(i)) - parr(idxn(j)) > 0.0)
                    auc += this_weight;
                       
            }
        }
        return_val = auc / sum_weights;
    """
    return scipy.weave.inline(code, ['idxp', 'idxn', 'parr', 'nidxn', 'nidxp', 'warr'],
            type_converters=scipy.weave.converters.blitz)


def auc_wmw_error(t, p, weights=None):
    """Returns 1.0 - AUC to mimic an error function
        (to pass into minimization routines)"""
    return 1.0 - auc_wmw_fast(t, p, weights)


def classification_error(y, ypred, thres=0.5, weights=None):
    """ y = 0, 1
    y pred = P(y == 1) is between 0 and 1
    Uses thres as the threshold
    y and ypred are numpy arrays
    weights = if provided is a y.shape() array with the weights
        take a weighted error in this case"""
    if weights is None:
        return ((ypred > thres).astype(np.int).reshape(-1, 1) != y.reshape(-1, 1)).sum() / float(len(y))
    else:
        return (((ypred > thres).astype(np.int).reshape(-1, 1) != y.reshape(-1, 1)) * weights.reshape(-1, 1)).sum() / float(weights.sum())


def precision_recall_f1(y, ypred, thres=0.5, weights=None):
    """y = 0/1 or -1/+1
    ypred = P(y == 1) is between 0 and 1
    y and ypred are numpy arrays
    weights = if provided is a y.shape() array with the weights
    take a weighted error in this case"""
    # need to properly handle case where y = (10, ), ypred=(10, 1)
    ypred_1 = (ypred > thres).reshape(-1, 1)
    yy = y.reshape(-1, 1)
    if weights is None:
        tp = np.sum(np.logical_and(ypred_1, yy == 1))
        fp = np.sum(np.logical_and(ypred_1, yy == 0))
        fn = np.sum(np.logical_and(~ypred_1, yy == 1))
    else:
        ww = weights.reshape(-1, 1)
        tp = np.sum(np.logical_and(ypred_1, yy == 1) * ww)
        fp = np.sum(np.logical_and(ypred_1, yy == 0) * ww)
        fn = np.sum(np.logical_and(~ypred_1, yy == 1) * ww)

    precision = tp / float(tp + fp)
    recall = tp / float(tp + fn)
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1








