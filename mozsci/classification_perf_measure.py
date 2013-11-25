"""
All the performance measures that we will be using for classification problems live here.
"""
import numpy as np


def classification_model_performance(observed, predicted, weight=None):
    """
    This is to check the performance of a classification algorithm.
    The observed values should be 0, 1, 2, etc. The weight is a list of the float numbers whose indices are
    the classes. For ex, if weight is [1, 5], then we have two classes in the classification problem. And
    the error caused by assigning class 0 instance to a class 1 instance is 1. The error caused by assigning
    a class 1 instance to a class 0 instance is 5.

    I like the returned perf measure to be in the range of [0, 1]. We should do so for at least the 'no-weight'
    case.

    Currently the value is, the lower, the better.
    """
    if weight is None:
        sum_incorrect = sum(observed != predicted)
    else:
        sum_incorrect = sum(weight[observed[ii]] for ii in xrange(len(observed)) if observed[ii] != predicted[ii])

    return sum_incorrect / float(len(predicted))


def classification_model_performance_matrix(observed, predicted):
    """
    This is to check the performance of a classification algorithm.
    The observed values should be 0, 1, 2, etc.

    We will use numpy's round number here - np.round(4.6) ( = 5.0). we can use int(np.round(4.6)) gives 5.
    """
    # assume that the classe categories start from 0.
    num_classes = int(max(observed)) + 1

    perf_2d_array = np.zeros([num_classes] * 2, dtype=int)

    for ii in xrange(len(observed)):
        # in case some algorithms return float numbers.
        predicted_class = int(np.round(predicted[ii]))
        perf_2d_array[observed[ii], predicted_class] += 1

    return perf_2d_array


def classification_model_performance_loss(observed, predicted, loss=None):
    """
    loss is a function with two inputs (i, j) where i is the real category and j is the predicted category.
    It returns a float number as the loss of assigning a category i instance to category j.
    A simple one is implemented as the default (see below in the function body).

    Another way to call this function is to define a loss function or lambda as below.
    classification_model_performance_loss(observed, predicted, loss=lambda i, j: (i-j)**2)
    """
    def default_loss(class_i, class_j):
        if class_i == class_j:
            return 0
        else:
            return 1

    if loss is None:
        loss = default_loss

    total_loss = sum(loss(observed[ii], int(np.round(predicted[ii]))) for ii in xrange(len(observed)))

    return total_loss
