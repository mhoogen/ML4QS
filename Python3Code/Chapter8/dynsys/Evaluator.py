##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

from inspyred.ec import emo
import random
from sklearn.metrics import mean_squared_error
import copy
import pandas as pd

# This class evaluates a dynamical systems model.
class Evaluator():

    training_data = {}
    test_data = {}
    model = []
    eval_aspects = []
    cleaned_eval_aspects = []
    default_start = 'self.'

    def __init__(self):
        self.training_data = {}
        self.test_data = {}
        self.model = []
        self.eval_aspects = []
        self.cleaned_eval_aspects = []

    # This sets the values for the training and test period. We will just
    # consider the eval_aspects (states/columns) for the evaluation.
    def set_values(self, m, train_X, train_y, test_X, test_y, eval_aspects):

        # Create a copy of the data
        self.training_data = copy.deepcopy(train_X)
        self.test_data = copy.deepcopy(test_X)

        # Add the targets we use from y to the copy of the data to create a single dataset
        # for training and testing.
        for col in eval_aspects:
            self.training_data[col[len(self.default_start):]] = train_y[col[len(self.default_start):]]
            self.test_data[col[len(self.default_start):]] = test_y[col[len(self.default_start):]]
        self.model = m
        self.eval_aspects = eval_aspects

        # The eval_aspects array is poluted with the 'self.', we also create a variant without.
        self.cleaned_eval_aspects = []
        for eval in eval_aspects:
            self.cleaned_eval_aspects.append(eval[len(self.default_start):])

    # This funtion generates a random initial candidate for our optimization algorithm and returns it.
    def generator(self, random, args):
        numb_parameters = len(self.model.parameter_names)
        return [random.uniform(-1.0, 1.0) for _ in range(numb_parameters)]

    # This functions takes a candidate (i.e. a number of parameter settings for our
    # dynamical systems model) and the dataset and evaluates how well it performs.
    # in terms of the mean squared error per eval_aspect. It return the fitness
    # and the prediction. If we have the per_time_step=True we will overwrite the
    # predicted values for the previous time point with the real values.
    def evaluator_internal(self, candidate, dataset, per_time_step=False):
        self.model.reset()
        y = []
        y.append(dataset.iloc[0,[dataset.columns.get_loc(x) for x in self.cleaned_eval_aspects]].values)

        # Go through the dataset, all but last as we need to evaluate our
        # prediction with the next time point.
        for step in range(0, len(dataset.index)-1):
            state_values = []

            # Get the relevant values for each of the states
            # in our model.
            for col in self.model.state_names:
                # Overwrite the values we predicted previously for the evaluation states
                # if we do it per time step or if we do not have any prediction yet.
                if per_time_step or (step == 0):
                    state_values.append(dataset.iloc[step, dataset.columns.get_loc(col[len(self.default_start):])])

                # Only overwrite values for the non eval states if we do not do it
                # per time step and use our predicted value for the eval aspects.
                else:
                    if col in self.eval_aspects:
                        state_values.append(pred_values[self.eval_aspects.index(col)])
                    else:
                        state_values.append(dataset.iloc[step, dataset.columns.get_loc(col[len(self.default_start):])])

            # Set the state values, parameter values, and execute the model.
            self.model.set_state_values(state_values)
            self.model.set_parameter_values(candidate)
            self.model.execute_steps(1)

            evals = []
            pred_values = []

            # Determine the error for the evaluation aspects.
            for eval in self.eval_aspects:
                pred_value = self.model.get_values(eval)[-1]
                pred_values.append(pred_value)
                mse = mean_squared_error([pred_value], [dataset.iloc[step+1, dataset.columns.get_loc(eval[len(self.default_start):])]])
                evals.append(mse)

            # Store the fitness for all aspects.
            fitness = emo.Pareto(evals)

            # Store the predicted values.
            y.append(pred_values)
        # And return the fitness and the predicted values.
        y_frame = pd.DataFrame(y, columns=self.cleaned_eval_aspects)
        return fitness, y_frame

    # This function evaluates a population of candidates in a multi-objective way.
    # It return the fitness on each eval_aspect for each candidate.
    def evaluator_multi_objective(self, candidates, args):
        fitness_values = []
        for c in candidates:
            fitness, y_pred = self.evaluator_internal(c, self.training_data, per_time_step=True)
            fitness_values.append(fitness)
        return fitness_values

    # This function evaluates a population of candidates in a single objective way.
    # It returns a single fitness value per candidate.
    def evaluator_single_objective(self, candidates, args):
        fitness_values = []
        for c in candidates:
            fitness, y_pred = self.evaluator_internal(c, self.training_data, per_time_step=True)

            # Sum the fitness values over all aspects.
            fitness_values.append(sum(fitness))
        return fitness_values

    # Generate a prediction for a candidate on either the training set (training=True) or the test
    # set. We can again select whether we want to set the values for the previous time point
    # to the true values all the time. We return the fitness van the predicted values.
    def predict(self, candidate, training=True, per_time_step=False):
        if training:
            fitness, y_pred = self.evaluator_internal(candidate, self.training_data, per_time_step=per_time_step)
        else:
            fitness, y_pred = self.evaluator_internal(candidate, self.test_data, per_time_step=per_time_step)
        return fitness, y_pred
