'''
This module can be used to glue together the model training process and prediction process in the possible 
production case. The codes are designed to be shared by these two processes as much as I can imagine so far.

Two things that highlight this design are 1. feature variable normalization. 2. variable definition support.
'''

import itertools
import json
import numpy as np
import pickle


logp1 = lambda x: np.log(x + 1)


def dumps_variable_def(variable_def_dict):
    """
    convert the useful information in the volume variable definitions to a dictionary for serialization.
    :return: the dictionary.
    """
    def return_jsonable_dict(variable_def):
        """
        This function filters out all fields that are not serializable by pickle or json.
        lambda or other function definitions are not json-able or pickle-able.
        :param variable_def:
        :return:
        """
        return dict((k, v) for k, v in variable_def.iteritems() if k != 'transform')

    ret = {
        'independent_variables': [return_jsonable_dict(x) for x in variable_def_dict['independent_variables']],
        'dependent_variable': return_jsonable_dict(variable_def_dict['dependent_variable']),
        'data_schema': [item['name'] for item in variable_def_dict['data_schema']]
    }

    return ret


class ModelDriver(object):
    """
    This class is used to drive any model/algorithm for training and prediction purposes. It's specifically
    designed so that we don't worry about the normalization for cross validation procedures.

    This is the class used to train and test one data set. -- supposed to be called once for each round in
    cross validations.

    The major goal is to make the normalization parameters as an output of the training process, and an
    input of the prediction process.
    """
    def __init__(self, variable_def=None, model=None):
        """
        The model must be a sklearn model or any models that has fit() and predict() as well as picklable.
        :param variable_def: the dictionary that contains the variable definition and normalization parameters.
        :param model: the sklearn model that defined outside. It will be trained here.
        """
        self.variable_def = variable_def
        self.model = model

    def serialize_model(self):
        """
        return a json string as the serialized model. The string should have both pickled model and normalization
        parameters.
        """
        variable_def = dumps_variable_def(self.variable_def)
        model = pickle.dumps(self.model)
        return json.dumps({'variable_def': variable_def, 'model': model})


    @classmethod
    def load_model(cls, model_json_str):
        """
        Load the serialized model. Afteer the loading, we can use pred method on new data sets.
        :param cls:
        :param model_json_str: a json string that contains the parameters.
        :return: the new object.
        """
        json_obj = json.loads(model_json_str)
        model = cls(variable_def=json_obj['variable_def'], model=pickle.loads(json_obj['model']))

        return model

    def normalize(self, X, calculate_mean_std=True):
        """
        normalize data before the training and predicting.
        :param X:
        :return:
        """
        assert(self.variable_def is not None)
        for index, item in enumerate(self.variable_def['independent_variables']):
            if item['normalization']:
                if calculate_mean_std:
                    m = np.mean(X[:, index])
                    s = np.std(X[:, index])
                    item['mean_std'] = [m, s]

                X[:, index] = (X[:, index] - item['mean_std'][0]) / item['mean_std'][1]

    def fit(self, predictors, y):
        """
        train the model using observations. Do normalization if needed.
        :param X: independent variables. 2-d numpy array.
        :param y: dependent variable. 1-d numpy array.
        :return: Nothing.
        """
        # we don't want to change the data in place, for cross-validation's reason.
        assert(self.model is not None)
        X = np.copy(predictors)
        self.normalize(X)

        self.model.fit(X, y)

    def predict(self, predictors, make_copy=True):
        """
        This method does the prediction using the model and saved normalization parameters.
        make_copy should be set to True if doing k-folds cv. It should be set to False for production.
        make_copy controls the data change in place.
        """
        if make_copy:
            predictors = np.copy(predictors)

        self.normalize(predictors, calculate_mean_std=False)

        return self.model.predict(predictors)


def get_training_data(fname, variable_def, delimiter='\t', header=True):
    """
    read data from fname, and returns a X, y. No normalization done here.
    """
    data = read_from_tsv(fname, mapping_input_line_2_numbers(variable_def), 
                         header=header, delimiter=delimiter)

    X = data[:, 1:]
    y = data[:, 0].reshape(-1)

    return X, y


def read_from_tsv(fname, mapping, header=True, delimiter='\t'):
    """
    Read data from a tsv file.
    :param mapping: the mapping from a list of fields to a list of numbers as features.
    :return: data in the form of a numpy array
    """
    data = []
    with open(fname) as fin:
        if header:
            fin.readline()

        for line in fin:
            splitted = line.strip().split(delimiter)
            ret = mapping(splitted)
            if ret is not None:
                data.append(ret)

    return np.array(data)


def mapping_input_line_2_numbers(variable_def):
    """
    Create a function that takes a splitted line of the training data file as the input.
    changed this to a higher order function so that we can use the variable definition as an argument.
    """
    def func(splitted):
        """
        :param splitted: a line of the training data that got splitted.
        :return: a list of feature variables.
        """
        return [item['transform'](splitted) for item in variable_def['data_schema']]

    return func


def k_fold_train_test_model(model, X, y, perf_measure, variable_def, k_fold=5):
    """
    running k-fold on a model and data set.
    :param model:
    :param X:
    :param y:
    :param perf_measure: A functions used to measure the performance of each of k-fold run. Its parameters are
                         the observed dependent variable and the predicted.
    :param k_fold:
    :return: the average of performance result for all the folds.
    """
    estimator = ModelDriver(copy.deepcopy(variable_def), model=model)

    def run_one_fold(train_indices, test_indices):
        estimator.fit(X[train_indices], y[train_indices])
        predicted = estimator.predict(X[test_indices])
        observed = y[test_indices]

        return perf_measure(observed, predicted)

    folds = cv_kfold(len(y), k_fold)

    perf_list = [run_one_fold(fold[0], fold[1]) for fold in folds]

    return np.mean(perf_list)

