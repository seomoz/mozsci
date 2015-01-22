from __future__ import absolute_import
from __future__ import print_function


# ensemble model selection
#
# based on "Ensemble Selection from Libraries of Models",
#  Caruana, Niculescu-Mizil, Crew, Ksikes
#  Proceedings of the 21st International Conference on ML, Banff Canada 2004
#

import numpy as np
import json
import six
from six.moves import range

class EnsembleModelSelector(object):
    """Implements
        "Ensemble Selection from Libraries of Models",
           Caruana, Niculescu-Mizil, Crew, Ksikes
           Proceedings of the 21st International Conference on ML, Banff Canada 2004

        Holds data
            .error = error function
            .ensemble = numpy array of the model weights
            .nmodels = the number of models added (=sum(ensemble)
            .ensemble_indices = the indices of models in ensemble included in final model

            .niter = number of iterations in an ensemble selction
            .nsort = the number of models to add at the beginning of the iteration

            For bagged selection:
                .nbags = number of bags to use for bagged 
                .pbags = the percent of models to use in each bag
           """

    def __init__(self, error=None, niter=10, nsort=5, nbags=20, pbags=0.5):
        """error = a callable thing (y, ypred) that computes the error
            it is minimized by the ensemble.
            needs to accept ypred that are averages of the individual model predictions

           niter = the number of iterations to use to add models
           nsort = the number of models added to start the ensemble (section 2.2)
           nbags = the number of bags to use for bagged selection
           pbags = the percentage of each models to include in each bag"""
        self.error = error
        self.ensemble = None
        self.ensemble_indices = None
        self.nmodels = 0
        self.niter = niter
        self.nsort = nsort
        self.nbags = nbags
        self.pbags = pbags


    def select_ensemble_bagged(self, y, ymodels, verbose=False):
        """Ensemble selection using bagged selection
        (Section 2.3 of the paper)"""
        ensemble = np.zeros(len(ymodels))
        indices = np.arange(len(ymodels))
        max_keep = int(self.pbags * len(ymodels))
        nmodels = 0
        for k in range(self.nbags):
            if verbose:
                print("Bagging number %s" % str(k+1))
            np.random.shuffle(indices)
            ymodels_bagged = np.array(ymodels)[indices[:max_keep]]
            self.select_ensemble(y, ymodels_bagged)
            # NOW self.ensemble is the selection of models in the bag
            # need to unroll these selected indices to those in the original ymodels
            ensemble[indices[:max_keep]] += self.ensemble
            nmodels += self.nmodels

        # set final ensemble
        self.ensemble = ensemble
        self.nmodels = nmodels
        self.ensemble_indices = np.arange(len(ymodels))[self.ensemble > 0.5]

    def select_ensemble(self, y, ymodels, early_termination = False):
        """Y = actual y = (N, ) numpy array
           ymodels = a list of predictions from different models.
             len(ymodels) = nmodels
             ymodels[k] = prediction for model k (N, ) numpy array
           DOESN'T do any bagging (section 2.3).  use select_ensemble_bagged"""
        # process:
        # (1) set the initial ensemble
        # (2) for each iteration, choose the model that decrease the error the most
        #     and update the current ensemble

        # (1)
        self.ensemble = np.zeros(len(ymodels))

        # do initial sort and insert these models into the ensemble
        # errors = a vector of errors corresponing to each model
        #   it will be updated for each iteration corresponding to the
        #   error for adding each model to the current ensemble
        errors = np.array([self.error(y, ypred) for ypred in ymodels])
        initial_models_to_add = errors.argsort()[0:self.nsort]
        self.ensemble[initial_models_to_add] = 1
        current_prediction = ymodels[initial_models_to_add[0]].astype(np.float)
        for i in initial_models_to_add[1:]:
            current_prediction += ymodels[i]
        current_prediction /= float(self.nsort)
        nmodels = self.nsort

        if early_termination: last_error = np.finfo(np.float).max
        # (2)
        for k in range(self.niter):
            # find the model that reduces error the most
            # current_prediction is averaged over nmodels
            # need to add in one more as a weighted average
            errors = np.array([self.error(y, current_prediction * (float(nmodels) / (nmodels + 1)) + ypred.astype(np.float) / float(nmodels + 1)) for ypred in ymodels])

            if early_termination:
                min_error = errors.min()
                if min_error < last_error: last_error = min_error
                else:break

            model_to_add = errors.argmin()

            self.ensemble[model_to_add] += 1
            current_prediction = current_prediction * (float(nmodels) / (nmodels + 1)) + ymodels[model_to_add].astype(np.float) / float(nmodels + 1)
            nmodels += 1

            print(("Iteration %s, error=%s" % (k, errors.min())))

        # pull out the indices of models included in the final ensemble
        self.ensemble_indices = np.arange(len(ymodels))[self.ensemble > 0.5]
        self.nmodels = nmodels


    def pred(self, ymodels):
        """Given the input from ymodels (same as input to select_ensemble),
        return the predicted probabilities"""
        pred = ymodels[self.ensemble_indices[0]] * self.ensemble[self.ensemble_indices[0]]
        for k in self.ensemble_indices[1:]:
            pred += ymodels[k] * self.ensemble[k]
        return pred.astype(np.float) / np.float(self.nmodels)

    def save_ensemble(self, fileout):
        """
        Serialize the ensemble.
        :param fileout: name of the file to write the json string, or a file object.
        :return: None
        """
        if self.ensemble is None or self.ensemble_indices is None:
            raise ValueError('The ensemble has not been properly trained.')

        model_json = {
            'nmodels': self.nmodels,
            'ensemble': self.ensemble[:].tolist(),
            'ensemble_indices': self.ensemble_indices[:].tolist(),
            }

        # save to the file
        if isinstance(fileout, six.string_types):
            with open(fileout, 'w') as f:
                json.dump(model_json, f)
        else:
            json.dump(model_json, fileout)

    @classmethod
    def load_ensemble(cls, model_json):
        """
        Load the serialized model. Afteer the loading, we can use pred method on new data sets.
        :param cls:
        :param model_json: name of the file to read in the json string, or a file object.
        :return: the new object.
        """
        if isinstance(model_json, six.string_types):
            with open(model_json, 'r') as f:
                model_json = json.load(f)

        ensemble = cls()
        ensemble.nmodels = model_json['nmodels']
        ensemble.ensemble = np.array(model_json['ensemble'], dtype = np.float64)
        ensemble.ensemble_indices = np.array(model_json['ensemble_indices'], dtype = np.int)

        return ensemble

if __name__ == "__main__":

    import pylab as plt
    from .evaluation import classification_error

    np.random.seed(2)

    # make the data
    N = 1000

    # some predictons
    # actual = 5 * x - 4 > 0
    x = np.linspace(0, 1, N)
    y = 5 * x - 4 > 0

    nmodels = 500
    ymodels = []
    for k in range(nmodels):
        m = np.random.rand(1) * 5 * (np.random.rand(N) - 0.5) + 5
        b = 3 * (np.random.rand(N) - 0.5) + 4
        thisy = (m * x - b > 0).astype(np.int)
        ymodels.append(thisy)

    ems = EnsembleModelSelector(classification_error, niter=25)
    ems.select_ensemble(y, ymodels)
    ypred = ems.pred(ymodels)
    classification_error(y, ypred)

    ems.select_ensemble_bagged(y, ymodels)
    ypred = ems.pred(ymodels)
    classification_error(y, ypred)


    fig = plt.figure(1)
    fig.clf()
    plt.scatter(x, y, marker='o', color='r')
    for k in range(40):
        plt.scatter(x, ymodels[k]+0.01 + k*0.01, marker='s', s=1, color='b')

    plt.scatter(x, ypred, marker='x', color='k')
    plt.plot(x, 0.5 * np.ones(x.shape), 'k')
    plt.plot(0.8 * np.ones((100, 1)), np.linspace(0, 1, 100), 'k')

    plt.title("Ensemble model selection via greedy sampling\nRed=actual, Blue=40 samples of noisy models, black=ensemble average")
    plt.xlabel("X")
    plt.ylabel("Y")
    fig.show()
    # fig.savefig("ensemble_model_average.png")




