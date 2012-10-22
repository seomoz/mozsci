"""Things to do cross validation"""

import numpy as np
from .map_train import TrainModelCV

def cv_kfold(ntrain, nk, seed=None):
    """k-fold cross validation

    ntrain = the integer number of training data points to sample
    nk = the number of splits of the training data
    optionally sets seed

    returns a list length nk.  Each element is a tuple:
        (train_indices, test_indices)

    NOTE: this is an approximate sampler, so the test set size
    isn't guaranteed to be 1 / nk, especially for small values of
    ntrain.
    """
    # need k probability splits 0-1

    # optionally set seed
    if seed is not None:
        np.random.seed(seed)

    # need k probability splits 0-1
    # the end points to sample
    fold_edges = np.linspace(0, 1, nk + 1)

    r = np.random.rand(ntrain)
    indices = np.arange(ntrain)
    folds = []
    for k in xrange(nk):
        folds.append(indices[np.logical_and(fold_edges[k] <= r, r < fold_edges[k + 1])])

    # make training + test arrays
    training_test = []
    for k in xrange(nk):
        training = []
        test = []
        for i in xrange(nk):
            if i != k:
                training.extend(folds[i])
            else:
                test.extend(folds[i])
        training_test.append([training, test])

    return training_test


def plot_cv_errors(errors, model, regparm, fignum):
    """Plots test vs training error for cross validation, as return from run_train_models

    errors = as returned from run_train_models
    model = a string with model name, e.g. "LogisticRegression"
    regparm = the name of regularization parameter, e.g. "lam"
    """
    import pylab as plt
    import re

    # accumulate the erorrs + the regularization parameters
    # errors_plot = [train, test] list
    errors_plot = []
    reg = []

    for desc, err in errors.iteritems():
        if re.search(model, desc):
            # it corresponds to this model
            # get the regularization parameter
            c = float(re.search("'%s':\s+([\.0-9-e]+)(}|,)" % regparm, desc).group(1))
            reg.append(c)
            errors_plot.append([err['train'], err['test']])

    errors_plot = np.asarray(errors_plot)
    reg = np.asarray(reg)
    plot_order = reg.argsort()

    fig = plt.figure(fignum)
    fig.clf()
    plt.plot(np.log(reg[plot_order]), errors_plot[plot_order, 0], label='train')
    plt.plot(np.log(reg[plot_order]), errors_plot[plot_order, 1], label='test')
    plt.legend()
    plt.grid(True)
    fig.show()


def learning_curves(model_description, X, y, kfolds=5, fignum=1):
    """Plot learning curves

    uses k-fold cross validation for 25, 50, 75, 100% of data
    
    model_description as input to TrainModelCV the model.  This defines
    the model to check.
    kfolds = use this many folds
    fignum = plot in this figure number
    """
    # use TrainModelCV to do so
    import pylab as plt

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    pct_data = np.array([0.25, 0.5, 0.75, 1.0])
    npct = len(pct_data)
    ndata = (pct_data * X.shape[0]).astype(np.int)
    test_errors = []
    train_errors = []
    for N in ndata:
        folds = cv_kfold(N, kfolds) 
        trainer = TrainModelCV(model_description,
                    X=X[indices][:N, :], y=y[indices][:N],
                    folds=folds)
        errors = trainer.run()
        test_errors.append(errors[errors.keys()[0]]['test'])
        train_errors.append(errors[errors.keys()[0]]['train'])

    fig = plt.figure(fignum)
    fig.clf()
    plt.plot(pct_data, train_errors, label='train')
    plt.plot(pct_data, test_errors, label='test')
    plt.xlabel("Percent of data")
    plt.legend()
    plt.ylabel(model_description[1].__name__)
    fig.show()


