"""
A few useful abstractions for input/output variables in machine learning
"""
import numpy as np
from itertools import izip


class Variable(object):
    """
    A Variable is one group of input or output to a model.
    """
    def __init__(self, name, transformer, ndim=1, ndimout=1):
        """
        name: the variable name
        transformer: implements the sklearn.Transformer API
            (fit, transform)
        ndim: the dimension of the variable (input)
        ndimout: the dimension of the output transform
        """
        self.name = name
        self.ndim = ndim
        self.ndimout = ndimout
        self._transformer = transformer

    # forwarding methods
    def fit(self, *args, **kwargs):
        return self._transformer.fit(*args, **kwargs)

    def transform(self, *args, **kwargs):
        return self._transformer.transform(*args, **kwargs)


class ModelVariables(object):
    """
    Hold sets of input and output variables for the model
    """
    def __init__(self, independent, dependent):
        """
        independent: list of Variable instances for the model input
        dependent: list of Variable instances for the model output
        """
        self.independent = independent
        self.dependent = dependent
        self.nin = sum([variable.ndim for variable in independent])
        self.nout = sum([variable.ndim for variable in dependent])


class ModelDriver(object):
    """
    This class is used to drive any model/algorithm for training and
    prediction purposes. It's specifically designed so that we don't need
    to worry about the normalization for cross validation procedures. It also
    supports the variable definitions that we use for data collection.
    """
    def __init__(self, variables, model):
        """
        variables: an instance of ModelVariables
        model: must implement the sklearn interface (fit, predict), as
            well as be picklable)
        """
        self.variables = variables
        self.model = model

    def _get_array(self, data, variables, dim):
        '''
        Get a numpy array from the data.
        '''
        if isinstance(data, np.ndarray):
            shape = data.shape
            if len(shape) == 2 and shape[1] == dim:
                # data is already an array
                return data

        # otherwise data should implement __getitem__
        first_var = variables[0]
        first_data = data[first_var.name]
        if isinstance(first_data, int) or isinstance(first_data, float):
            ret = np.zeros((1, dim))
        else:
            ret = np.zeros((len(first_data), dim))
        ind = first_var.ndim
        if ind == 1:
            ret[:, 0] = first_data
        else:
            ret[:, :ind] = first_data
        for variable in variables[1:]:
            if variable.ndim == 1:
                ret[:, ind] = data[variable.name]
            else:
                ret[:, ind:(ind + variable.ndim)] = data[variable.name]
            ind += variable.ndim
        return ret

    def _transform(self, X, variables, fit=False):
        '''
        Transform the data
        '''
        # get the output dimensions
        try:
            ndimout = [v.ndimout for v in variables]
        except AttributeError:
            ndimout = []
            for v in variables:
                if hasattr(v, 'ndimout'):
                    ndimout.append(v.ndimout)
                else:
                    ndimout.append(1)
        nout = sum(ndimout)
        ret = np.zeros((len(X), nout))
        ind = 0
        indout = 0
        for variable, dimout in izip(variables, ndimout):
            if fit:
                variable.fit(X[:, ind:(ind + variable.ndim)])
            ret[:, indout:(indout + dimout)] = variable.transform(
                X[:, ind:(ind + variable.ndim)])
            ind += variable.ndim
            indout += dimout
        return ret

    def fit(self, predictors, y):
        """
        train the model using observations.
        :param X: independent variables. 2-d numpy array or something
            implementing __getitem__
        :param y: dependent variable. numpy array or something implementing
            __getitem__
        :return: Nothing.
        """
        # (1) get predictor, predicted array
        X = self._get_array(predictors, self.variables.independent,
            self.variables.nin)
        yy = self._get_array(y, self.variables.dependent,
            self.variables.nout)

        # (2) fit transforms
        XX = self._transform(X, self.variables.independent, True)
        YY = self._transform(yy, self.variables.dependent, True)

        # (3) fit the model
        self.model.fit(XX, YY)

    def predict(self, predictors, predict_prob=False):
        """
        This method does the prediction using the model and saved
        normalization parameters.
        """
        X = self._get_array(predictors, self.variables.independent,
            self.variables.nin)
        XX = self._transform(X, self.variables.independent, False)

        if predict_prob:
            return self.model.predict_proba(XX)
        else:
            return self.model.predict(XX)

    def dumps(self):
        '''
        Return a string representation of this instance
        '''
        import cPickle
        return cPickle.dumps(self)

    @classmethod
    def loads(cls, string):
        '''
        Return an instance from the serialized string
        '''
        import cPickle
        return cPickle.loads(string)

