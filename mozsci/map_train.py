"""Train models in parallel"""
from __future__ import absolute_import

import numpy as np
from six.moves import range

class TrainModelCV(object):
    def __init__(self,
                 model_description=[None, None, '', (), {}],
                 X=None, y=None, Xtest=None, ytest=None,
                 folds=None, weights=None, weightstest=None, fit_kwargs={}):
        """
        model_description = [model_init, error, model_save_file, args, kwargs]
                WHERE
            model_init = a callable thing model_init(args, kwargs) that returns a model
                object.  This has an interface as follows:
                   model.fit(X, y) = trains
                   model.predict(X) = predicts
                   model.save_model(filename) = serializes model to a file
            error = a callable thing that computes error as error(Yactual, Ypred)
            model_save_file = if provided, then saves the model to this file
            args, kwards = passed to model_init(*args, **kwargs)

            fit_kwargs = anything to pass down the model.fit routine (error tolerance, etc)

        X, y = training dataset (required)
        Xtest, Ytest = testing dataset (if provided, then computes error on this dataset

        folds = if provided, then gives a set of splits to use for k-fold cross validation.
        folds is a length-k list.  Each element of the list is a tuple, where the first
        element of the tuple gives the training indices, the second the test indices.
        folds can easily be generated with a call to cv_kfold.

        If doing a k-fold CV, then Xtest and ytest are ignored, and X and y are split
        (and an error is raised if Xtest and ytest are provided).
        The model_save_file is also ignored in this case
        The errors data structure reports the average error for each fold.
        """
        self.model_description = model_description
        self.model_init = model_description[0]
        self.error = model_description[1]
        self.model_save_file = model_description[2]
        self.X = X
        self.y = y
        self.Xtest = Xtest
        self.ytest = ytest
        self.folds = folds
        if folds is not None:
            assert Xtest is None and ytest is None
        self.weights = weights
        self.weightstest = weightstest
        self._fit_kwargs = fit_kwargs


    def run(self):
        if self.folds is not None:
            errors = self._run_kfold()
        else:
            errors, model = self._run_one_train_test(self.X, self.y, self.Xtest, self.ytest, self.weights, self.weightstest, fit_kwargs=self._fit_kwargs)

            # save to file if needed
            if self.model_save_file is not None:
                model.save_model(self.model_save_file)

        # prepare errors for output
        errors_ret = {}
        errors_ret[str(self.model_description)] = errors

        return errors_ret


    def _run_kfold(self):
        # do k-fold cross validation
        errors = []
        for k in range(len(self.folds)):
            train_indices = self.folds[k][0]
            test_indices = self.folds[k][1]

            if self.weights is None:
                this_error, model = self._run_one_train_test(self.X[train_indices, :], self.y[train_indices], self.X[test_indices, :], self.y[test_indices], fit_kwargs=self._fit_kwargs)
            else:
                this_error, model = self._run_one_train_test(self.X[train_indices, :], self.y[train_indices], self.X[test_indices, :], self.y[test_indices], self.weights[train_indices], self.weights[test_indices], fit_kwargs=self._fit_kwargs)

            errors.append(this_error)

        # return average error
        # for aggregate error functions, can return
        #   errors['train'] = {'error1': 0.5, 'error2': 0.2}, ...
        # also support this case
        ret = {}
        if type(errors[0]['train']) == dict:
            for k in ['train', 'test']:
                ret[k] = {}
                for error_type in list(errors[0]['train'].keys()):
                    ret[k][error_type] = np.mean([ele[k][error_type] for ele in errors])
        else:
            for k in ['train', 'test']:
                ret[k] = np.mean([ele[k] for ele in errors])

        return ret


    def _run_one_train_test(self, X, y, Xtest, ytest, weights=None, weightstest=None, fit_kwargs={}):
        # initialize model
        # train
        # compute error

        # initialize
        model = self.model_init(*self.model_description[3], **self.model_description[4])

        # train
        try:
            model.fit(X, y, weights=weights, **fit_kwargs)
        except TypeError:   # model doesn't do weighted learning
            model.fit(X, y, **fit_kwargs)

        # compute error
        errors = {}
        ypred = model.predict(X)
        if weights is None:
            errors['train'] = self.error(y, ypred)
        else:
            errors['train'] = self.error(y, ypred, weights=weights)


        if Xtest is not None:
            ypred = model.predict(Xtest)
            if weightstest is None:
                errors['test'] = self.error(ytest, ypred)
            else:
                errors['test'] = self.error(ytest, ypred, weights=weightstest)
        else:
            errors['test'] = None

        return errors, model


def _pool_helper(model_description, X=None, y=None, Xtest=None, ytest=None,
                 folds=None, weights=None, weightstest=None):
    # a helper for Pool class.
    # this creates an instance of TrainModelCV and runs it
    trainer = TrainModelCV(model_description,
                X=X, y=y, Xtest=Xtest, ytest=ytest,
                folds=folds, weights=weights, weightstest=weightstest)
    return trainer.run()



def run_train_models(processes, model_library, **kwargs):
    """Train many supervised learning problems in parallel

    model_library = a list specifying the model library for the dataset in
            format needed for TrainModelCV
            **kwargs: all the rest of the input to TrainModelCV"""
    # sample input for model_library:
    #          [[LogisticRegression, classification_error, 'parameters.json', (), {'lam':0.5}],
    #          [LogisticRegression, auc_wmw_fast, None, (), {'C':50}]]


    if processes > 1:

        # use a process pool top execute all the training jobs
        # collect the results and combine to return
        from multiprocessing import Pool

        p = Pool(processes)

        results = []
        for model in model_library:
            results.append(p.apply_async(_pool_helper, (model, ), kwargs))

        # wait on the pool to finish
        p.close()
        p.join()

        # collect the results
        ret = {}
        for result in results:
            ret.update(result.get())

    else:
        # don't need a pool
        ret = {}
        for model in model_library:
            ret.update(_pool_helper(model, **kwargs))

    return ret


