"""
This new version of Variables and Model training codes are not compatible with the first version, which was used
for volume estimation etc.

I plan to rename this module to a different name and keep both version to avoid the rewriting of the modules that
are dependent on the first version of variables.
"""
import numpy as np
import pickle
import json
import copy
from mozsci.cross_validate import cv_kfold


class Variable(object):
    """
    This is the base class for one single variable.

    Any variable can be an independent variable, dependent variable or just an input variable from which we get
    one or multiple dependent/independent variables.
    """
    def __init__(self, name=None, pre_transform=None, transform=None, post_transform=None,
                 normalization=False, mean_std=None, description=None):
        """
        pre_transform: the transform used to collect data set for training.
        transform: the transform used to create variables used in training algorithms.
        post_transform: the transform used to get variables used to run on the trained model.

        To be more specific, pre_transform is most likely used in training data collection.
        transform's output is a feature variable (dependent or independent) in the machine linear algorithm.
        post_transform's input is like pre_transform, but output is like the output of transform. It is used in
            prediction stage, very likely in production.

        normalization controls if this variable needs to be normalized in the training and prediction processes.
        mean_std is used for the normalization.

        So it's possible some variable has None as one of its tranfoms.
        """
        self.name = name
        self.pre_transform = pre_transform
        self.transform = transform
        self.post_transform = post_transform
        self.normalization = normalization
        self.mean_std = mean_std
        self.description = description

    def __str__(self):
        return 'name: %s, normalization: %s, mean_std: %s, description: %s' % \
               (self.name, self.normalization, self.mean_std, self.description)

    def dump_parameters(self):
        """
        used for serialization. We only need to serialize the parameters that are not known at the time when the
        variable is created. And we cannot (easily and cleanly) serialize the transforms anyway.
        based on json.
        """
        return json.dumps({'normalization': self.normalization, 'mean_std': self.mean_std})

    def load_parameters(self, var_json_str):
        """
        load the serialized variable and return an object.
        """
        json_obj = json.loads(var_json_str)

        self.normalization = json_obj['normalization']
        self.mean_std = json_obj['mean_std']


class ModelVariables(object):
    """
    This is the class that defines all the feature variables used in the model and (possibly) some other variables
    used in the data collection stage.
    """
    def __init__(self, independent, dependent, schema):
        """
        each of these is a list of Variable objects that were defined in previous class.
        independent variables and dependent variables are used in the model training and prediction.
        schema is used in the training data creation. The main purpose is to create a string to write to a
        (training data) file. And then this file will be used as the input for model training step.

        Most of times the dependent variable should be a list of one variable, but it is possible to have
        multiple columns of dependent variables (ex. 1 from n classification).
        """
        self.independent = independent
        self.dependent = dependent
        self.schema = schema

    def __str__(self):
        """
        return a string format of the variables.
        """
        def print_variables(vars):
            return '; '.join(str(variable) for variable in vars)

        return 'indepdent: [%s], dependent: [%s], schema [%s]' % \
               (print_variables(self.independent),
                print_variables(self.dependent),
                print_variables(self.schema))

    def data_str(self, input_data, delimiter='\t'):
        """
        transform the source data into a strings for writing to a training data file. Schema is the format.
        This method assumes all the variables defined in the schema use the same input as the argument.
        """
        return delimiter.join(str(variable.pre_transform(input_data)) for variable in self.schema)

    def write_training_data(self, data, output, header=False, delimiter='\t'):
        """
        If given an iterator of training data source data set, we can use this method to write all the data out
        to a file.
        """
        with open(output, 'w') as fout:
            if header:
                fout.write(delimiter.join(variable.name for variable in self.schema))
                fout.write('\n')

            for item in data:
                fout.write('%s\n', self.data_str(item, delimiter))

    def dump_parameters(self):
        """
        used for serialization.
        based on json.
        """
        def dump_variables(variables):
            return json.dumps([variable.dump_parameters() for variable in variables])

        return json.dumps({'independent': dump_variables(self.independent),
                           'dependent': dump_variables(self.dependent),
                           'schema': dump_variables(self.schema)})

    def load_parameters(self, var_json_str):
        """
        load the serialized variable parameters for normalization.
        """
        def loads_variables(variables, json_str):
            serialized_variables = json.loads(json_str)
            assert len(variables) == len(serialized_variables)
            for i in xrange(len(variables)):
                variables[i].load_parameters(serialized_variables[i])

        json_obj = json.loads(var_json_str)
        loads_variables(self.independent, json_obj['independent'])
        loads_variables(self.dependent, json_obj['dependent'])
        loads_variables(self.schema, json_obj['schema'])

    def get_training_data(self, fname, delimiter='\t', header=True):
        """
        read data from fname, and returns a X, y. No normalization done here.
        """
        data = self._read_from_tsv(fname, header=header, delimiter=delimiter)

        dependent_dim = len(self.dependent)

        X = data[:, dependent_dim:]

        if dependent_dim == 1:
            y = data[:, 0].reshape(-1)
        else:
            y = data[:, 0:dependent_dim]

        return X, y

    def _read_from_tsv(self, fname, header=True, delimiter='\t'):
        """
        Read data from a tsv file.
        :param mapping: the mapping from a list of fields to a list of numbers as features.
        :return: data in the form of a numpy array
        """
        data = []
        mapping = self._mapping_input_line_2_numbers()
        with open(fname) as fin:
            if header:
                fin.readline()

            for line in fin:
                splitted = line.strip().split(delimiter)
                ret = mapping(splitted)
                if ret is not None:
                    data.append(ret)

        return np.array(data)

    def _mapping_input_line_2_numbers(self):
        """
        Create a function that takes a splitted line of the training data file as the input.
        changed this to a higher order function so that we can use the variable definition as an argument.

        Note: this function is to create all the independent and dependent variables used in machine learning
        model training step.
        """
        def func(splitted):
            """
            :param splitted: a line of the training data that got splitted.
            :return: a list of feature variables.
            """
            # We first create dependent variables and then we create independent variables.
            return [item.transform(splitted) for item in self.dependent + self.independent]

        return func


class ModelDriver(object):
    """
    This class is used to drive any model/algorithm for training and prediction purposes. It's specifically
    designed so that we don't need to worry about the normalization for cross validation procedures. It also
    supports the variable definitions that we use for data collection.

    This is the class used to train and test one data set. -- it is supposed to be called once for each round in
    cross validations.

    The major goal is to make the normalization parameters as an output of the training process, and an
    input of the prediction process.
    """
    def __init__(self, variable_def=None, model=None):
        """
        The model must be a sklearn model or any models that has fit() and predict() as well as picklable.
        :param variable_def: an object of ModelVariables that contains the variable definition and normalization
                             parameters.
        :param model: the sklearn model that defined outside. It will be trained here.

        """
        self.variable_def = variable_def
        self.model = model

    def serialize_model_parameters(self):
        """
        return a json string as the serialized model. The string should have both pickled model and normalization
        parameters.
        """
        return json.dumps({'model': pickle.dumps(self.model), 'variable_def': self.variable_def.dump_parameters()})

    def load_model_parameters(self, model_json_str):
        """
        Load the serialized model. Afteer the loading, we can use pred method on new data sets.
        :param cls:
        :param model_json_str: a json string that contains the parameters.
        :return: the new object.
        """
        json_obj = json.loads(model_json_str)
        self.variable_def.load_parameters(json_obj['variable_def'])
        self.model = pickle.loads(json_obj['model'])

    def normalize(self, X, calculate_mean_std=True):
        """
        normalize data before the training and predicting.
        :param X:
        :return:
        """
        for index, item in enumerate(self.variable_def.independent):
            if item.normalization:
                if calculate_mean_std:
                    m = np.mean(X[:, index])
                    s = np.std(X[:, index])
                    item.mean_std = [m, s]

                X[:, index] = (X[:, index] - item.mean_std[0]) / item.mean_std[1]

        # print 'normalization is done.'
        # rows, columns = X.shape
        # for i in range(columns):
        #     print 'asserting ... ', i
        #     assert not np.any(np.isnan(X[:, i]) | np.isinf(X[:, i]))

        # print 'assert succeeded.'

    def fit(self, predictors, y):
        """
        train the model using observations. Do normalization if needed.
        :param X: independent variables. 2-d numpy array.
        :param y: dependent variable. 1-d numpy array.
        :return: Nothing.
        """
        # we don't want to change the data in place, for cross-validation's reason.
        X = np.copy(predictors)
        self.normalize(X)

        self.model.fit(X, y)

    def predict(self, predictors, make_copy=True, predict_prob=False):
        """
        This method does the prediction using the model and saved normalization parameters.
        make_copy should be set to True if doing k-folds cv. It should be set to False for production.
        make_copy controls the data change in place.
        """
        if make_copy:
            predictors = np.copy(predictors)

        self.normalize(predictors, calculate_mean_std=False)

        if predict_prob:
            return self.model.predict_proba(predictors)
        else:
            return self.model.predict(predictors)

    def item_predict(self, item, predict_prob=False):
        """
        This will mostly likely be used in production. It first creates a numpy array based on the post-transforms.
        And then feed this numpy array to predict method.
        """
        X_lst = [variable.post_transform(item) for variable in self.variable_def.independent]
        X = np.array(X_lst).reshape((1, -1))

        return self.predict(X, make_copy=False, predict_prob=predict_prob)


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


