import numpy as np
import pickle
import json
import copy
from mozsci.cross_validate import cv_kfold

logp1 = lambda x: np.log(x + 1)


def get_float(num_str):
    if num_str == 'None':
        return 0.0
    else:
        return float(num_str)

# dependent variable.
get_log_google_volume = lambda x: logp1(get_float(x[2]))

# independent variables.
get_log_bing_volume = lambda x: logp1(get_float(x[4]))
get_fwe_volume = lambda x: get_float(x[3]) ** 0.2
get_bing_volume_boolean = lambda x: x[4] == 'None'

# GrepWord data. starts from column 14 (the 15th column).
grep_words_start_index = 14
get_grep_word_boolean = lambda x: x[grep_words_start_index] == '0' and x[grep_words_start_index + 1] == '0' and \
                                  x[grep_words_start_index + 2] == '0' and x[grep_words_start_index + 3] == '0'
get_grep_word_cmp = lambda x: get_float(x[grep_words_start_index])   # seems that they are in [0, 1] range. -- competition score.
get_grep_word_gms = lambda x: logp1(get_float(x[grep_words_start_index + 1]))   # global monthly search volume
get_grep_word_lms = lambda x: logp1(get_float(x[grep_words_start_index + 2]))   # local monthly search volume
get_grep_word_cpc = lambda x: logp1(get_float(x[grep_words_start_index + 3]))   # cost per click
get_grep_word_3month = lambda x: logp1(get_float(x[grep_words_start_index + 4]))   # 3 month data. Currently for 04/2013 - 06/2013.

# SemRush data. starts from column 19 (the 20th column).
sem_rush_start_index = 19
get_sem_rush_boolean = lambda x: x[sem_rush_start_index] == '0' and x[sem_rush_start_index + 1] == '0' and \
                                 x[sem_rush_start_index + 2] == '0' and x[sem_rush_start_index + 3] == '0'
get_sem_rush_number_results = lambda x: logp1(get_float(x[sem_rush_start_index]))
get_sem_rush_cpc = lambda x: get_float(x[sem_rush_start_index + 1])
get_sem_rush_competition = lambda x: get_float(x[sem_rush_start_index + 2])
get_sem_rush_volume = lambda x: logp1(get_float(x[sem_rush_start_index + 3]))


# definition of dependent and independent variables.
dependent_variable = {'name': 'google_volume', 'transform': get_log_google_volume, 'normalization': False}

# Bing and Fwe Data.
log_bing_volume = {'name': 'bing_log_volume', 'transform': get_log_bing_volume, 'normalization': True}
log_fwe_volume = {'name': 'fwe_log_volume', 'transform': get_fwe_volume, 'normalization': True}
bing_volume_boolean = {'name': 'bing_volume_boolean', 'transform': get_bing_volume_boolean, 'normalization': False}

# GrepWord variables.
grep_words_boolean = {'name': 'grep_words_boolean', 'transform': get_grep_word_boolean, 'normalization': False}
grep_words_cmp = {'name': 'grep_words_cmp', 'transform': get_grep_word_cmp, 'normalization': False}
# grep_words_gms is now replaced by the 3 month data.
# grep_words_gms = {'name': 'grep_words_gms', 'transform': get_grep_word_gms, 'normalization': True}
grep_words_lms = {'name': 'grep_words_lms', 'transform': get_grep_word_lms, 'normalization': True}
grep_words_cpc = {'name': 'grep_words_cpc', 'transform': get_grep_word_cpc, 'normalization': True}
grep_words_3month = {'name': 'grep_words_3month', 'transform': get_grep_word_3month, 'normalization': True}

# SemRush variables.
sem_rush_boolean = {'name': 'sem_rush_boolean', 'transform': get_sem_rush_boolean, 'normalization': False}
sem_rush_number_results = {'name': 'sem_rush_number_results', 'transform': get_sem_rush_number_results, 'normalization': True}
sem_rush_cpc = {'name': 'sem_rush_cpc', 'transform': get_sem_rush_cpc, 'normalization': True}
sem_rush_competition = {'name': 'sem_rush_competition', 'transform': get_sem_rush_competition, 'normalization': True}
sem_rush_volume = {'name': 'sem_rush_volume', 'transform': get_sem_rush_volume, 'normalization': True}

independent_variables = [log_bing_volume, log_fwe_volume, bing_volume_boolean, grep_words_boolean,
                         grep_words_cmp, grep_words_lms, grep_words_cpc, grep_words_3month]

# use SEMrush to replace grep words.
# independent_variables = [log_bing_volume, log_fwe_volume, bing_volume_boolean, sem_rush_boolean,
#                          sem_rush_number_results, sem_rush_cpc, sem_rush_competition, sem_rush_volume]

# It seems that the data_schema is the schema used in creating the training data set.
data_schema = [dependent_variable] + independent_variables

## The following is what we need.
volume_estimate_variable_def = {'independent_variables': independent_variables,
                                'dependent_variable': dependent_variable,
                                'data_schema':  data_schema}


class VariableDefinition(object):
    """
    This is the class to define data that will be used for data collection, model training, and model prediction
    phase.

    This class only supports 2d matrix data sets. It does not support 3d (such as listwise data) or higher
    dimensional data sets.

    We don't necessarily define this class, because it is simply a dictionary.
    Each variable can have as many as three transformations defined:
    1. used for data creation -> transform (input) into data values to be written in the training data file.
    2. used for model training, ie. to create feature variables. transform (each line in the training data file)
       into a feature variable.
    3. used for model prediction. transform (input) to a feature variable that will be used to predict by the
       serialized model.

    """

    def __init__(self, variables):
        """
        variables is the dictionary that defines all the transforms, names, descriptions and normalizations for
        every variable.
        """
        assert not variables is None
        self.variables = variables

    def dumps_variable_def(self):
        """
        convert the useful information in the volume variable definitions to a dictionary for serialization.
        :return: the dictionary.

        One example of the variable_def_dict is as follows,
        bing_volume = {'name': 'bing_volume',
                       'transform': lambda splitted_line: int(splitted_line[2]),
                       'normalization': True}
        log_fwe_volume = {'name': 'fwe_log_volume',
                          'transform': lambda splitted_line: float(splittxed_line[3]),
                          'normalization': True}
        log_google_volume = {'name': 'google_volume',
                             'transform': lambda splitted_line: float(splittxed_line[0]),
                             'normalization': False}

        variable_definition_example = {'independent_variables': [log_bing_volume, log_fwe_volume],
                                        'dependent_variable': log_google_volume,
                                        'data_schema': [dependent_variable] + independent_variables}

        The 'transform' in each variable dictionary defines how we collect data from a tsv file.


        Note (12022013): I am not sure if this serialization is still needed, because we can simply use the
        shared object of class VariableDefinition in different phases of the model.
        """
        def return_jsonable_dict(variable_def):
            """
            This function filters out all fields that are not serializable by pickle or json.
            lambda or other function definitions are not json-able or pickle-able.
            :param variable_def:
            """
            serializable_fields = set(['name', 'normalization', 'description'])
            return dict((k, v) for k, v in variable_def.iteritems() if k in serializable_fields)

        ret = {
            'independent_variables': [return_jsonable_dict(x) for x in self.variables['independent_variables']],
            'dependent_variable': return_jsonable_dict(self.variables['dependent_variable']),
            'data_schema': [item['name'] for item in self.variables['data_schema']]
        }

        return ret


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
        variable_def = self.variable_def.dumps_variable_def()
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
