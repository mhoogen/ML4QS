##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

import pandas as pd
import scipy.linalg
import copy
import random
import numpy as np
from scipy import linalg
import inspyred
from Chapter8.dynsys.Model import Model
from Chapter8.dynsys.Evaluator import Evaluator
from pybrain.structure import RecurrentNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, FullConnection
from pybrain.datasets import SequentialDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.supervised.trainers import RPropMinusTrainer
from pybrain.tools.validation import testOnSequenceData
from pybrain.tools.shortcuts import buildNetwork
from Chapter7.Evaluation import ClassificationEvaluation
from Chapter7.Evaluation import RegressionEvaluation
import sys
import matplotlib.pyplot as plot
import pyflux as pf
from statsmodels.tsa.arima_model import ARIMA



# The class includes several algorithms that capture the temporal dimension explicitly for
# classification problems.
class TemporalClassificationAlgorithms:

    # This function converts a single dataset (no test or train split up) with possibly
    # categorical attributes to a numerical dataset, where categorical attributes are
    # taken as dummy variables (i.e. binary columns for each possible value).
    def create_numerical_single_dataset(self, dataset):
        return copy.deepcopy(pd.get_dummies(pd.DataFrame(dataset), prefix='', prefix_sep=''))

    # This function converts a train and test dataset with possibly
    # categorical attributes to a numerical dataset, where categorical attributes are
    # taken as dummy variables (i.e. binary columns for each possible value).
    def create_numerical_multiple_dataset(self, train, test):

        # Combine the two datasets as we want to include all possible values
        # for the categorical attribute.
        total_dataset = train.append(test)

        # Convert and split up again.
        total_dataset = pd.get_dummies(pd.DataFrame(total_dataset), prefix='', prefix_sep='')
        new_train = copy.deepcopy(total_dataset.iloc[0:len(train.index),:])
        new_test = copy.deepcopy(total_dataset.iloc[len(train.index):len(train.index)+len(test.index),:])
        return new_train, new_test

    # This function initializes an echo state network given the specified number of
    # inputs, outputs, and nodes in the reservoir. It returns the weight matrices W_in,
    # W, and W_back.
    def initialize_echo_state_network(self, inputs, outputs, reservoir):

        # http://minds.jacobs-university.de/mantas/code
        # Create random matrices.
        Win = (np.random.rand(reservoir,1+inputs)-0.5) * 1
        W = np.random.rand(reservoir,reservoir)-0.5
        Wback = (np.random.rand(reservoir,outputs)-0.5) * 1

        # Adjust W to "guarantee" the echo state property.
        rhoW = max(abs(linalg.eig(W)[0]))
        W *= 1.25 / rhoW
        return Win, W, Wback

    # Predict the values of an echo state network given the matrices Win, W, Wback, Wout, the setting for a,
    # the reservoir size, and the dataset (which potentially includes the target as well). The cols are
    # the relevant columns of X. Finally, per_time_step=True means that we feed to correct output back into
    # the network instead of our prediction (this requires a non empty y_true). It returns the predicted class
    # and probabilites per class in the form a pandas dataframe with a column per class value.
    # http://minds.jacobs-university.de/sites/default/files/uploads/mantas/code/minimalESN.py.txt

    def predict_values_echo_state_network(self, Win, W, Wback, Wout, a, reservoir_size, X, y_true, cols, per_time_step):
        # http://minds.jacobs-university.de/mantas/code
        # Set the initial activation to zero.
        x = np.zeros((reservoir_size,1))
        Y = []

        # Predict all time points.
        for t in range(0, len(X.index)):

            # Set the input according to X.
            u = X.iloc[t,:].values

            # If we have a previous time point
            if t > 0:

                # If we predict per time step, set the previous value
                # to the true previous value.
                if per_time_step:
                    y_prev = y_true.iloc[t-1,:].values

                # Otherwise set it to the predicted value.
                else:
                    y_prev = y

            # If we do not have a previous time point, set the values to 0.
            else:
                y_prev = np.array([0]*len(cols))

            # Compute the activation of the reservoir.
            x = (1-a)*x + a*np.tanh( np.dot( Win, np.vstack(np.insert(u,0,1)) ) + np.dot( W, x ) + np.dot( Wback, np.vstack(y_prev) ))

            # And the output.
            y = np.tanh( np.dot( Wout, np.hstack(np.insert(np.insert(x, 0, u), 0, 1)) ))
            Y.append(y)
        y_result = pd.DataFrame(Y, columns=cols, index=X.index)
        return y_result.idxmax(axis=1), y_result

    # Given a dictionary with an ordered list of parameter values to try, return
    # all possible combinations in the form of a list.
    def generate_parameter_combinations(self, parameter_dict, params):
        combinations = []
        if len(params) == 1:
            values =  parameter_dict[params[0]]
            for val in values:
                combinations.append([val])
            return combinations
        else:
            params_without_first_element = copy.deepcopy(list(params))
            params_without_first_element.pop(0)
            params_without_first_element_combinations = self.generate_parameter_combinations(parameter_dict, params_without_first_element)
            values_first_element = parameter_dict[list(params)[0]]
            for i in range(0, len(values_first_element)):
                for j in range(0, len(params_without_first_element_combinations)):
                    list_obj = [values_first_element[i]]
                    list_obj.extend(params_without_first_element_combinations[j])
                    combinations.append(list_obj)
            return combinations

    def gridsearch_reservoir_computing(self, train_X, train_y, test_X, test_y, per_time_step=False, error = 'mse', gridsearch_training_frac=0.7):
        tuned_parameters = {'a': [0.6, 0.8], 'reservoir_size':[400, 700, 1000]}
#        tuned_parameters = {'a': [0.4], 'reservoir_size':[250]}
        params = tuned_parameters.keys()
        combinations = self.generate_parameter_combinations(tuned_parameters, params)
        split_point = int(gridsearch_training_frac * len(train_X.index))
        train_params_X = train_X.iloc[0:split_point,]
        test_params_X = train_X.iloc[split_point:len(train_X.index),]
        train_params_y = train_y.iloc[0:split_point,]
        test_params_y = train_y.iloc[split_point:len(train_X.index),]

        if error == 'mse':
            best_error = sys.float_info.max
        elif error == 'accuracy':
            best_error = 0

        best_combination = []
        for comb in combinations:
            print(comb)
            # Order of the keys might have changed.
            keys = list(tuned_parameters.keys())
            pred_train_y, pred_test_y, pred_train_y_prob, pred_test_y_prob = self.reservoir_computing(train_params_X, train_params_y, test_params_X, test_params_y,
                                                                                                     reservoir_size=comb[keys.index('reservoir_size')], a=comb[keys.index('a')], per_time_step=per_time_step,
                                                                                                     gridsearch=False)

            if error == 'mse':
                eval = RegressionEvaluation()
                mse = eval.mean_squared_error(test_params_y, pred_test_y_prob)
                if mse < best_error:
                    best_error = mse
                    best_combination = comb
            elif error == 'accuracy':
                eval = ClassificationEvaluation()
                acc = eval.accuracy(test_params_y, pred_test_y)
                if acc > best_error:
                    best_error = acc
                    best_combination = comb

        print('-------')
        print(best_combination)
        print('-------')
        return best_combination[keys.index('reservoir_size')], best_combination[keys.index('a')]

    def normalize(self, train, test, range_min, range_max):

        total = copy.deepcopy(train).append(test, ignore_index=True)

        max = total.max()
        min = total.min()
        difference = max - min
        difference = difference.replace(0, 1)

        new_train = (((train - min)/difference) * (range_max - range_min)) + range_min
        new_test = (((test - min)/difference) * (range_max - range_min)) + range_min
        return new_train, new_test, min, max


    def denormalize(self, y, min, max, range_min, range_max):
        difference = max - min
        difference = difference.replace(0, 1)

        y = (y - range_min)/(range_max - range_min)

        return (y * difference) + min

    # Apply an echo state network for classification upon the training data (with the specified reservoir size),
    # and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def reservoir_computing(self, train_X, train_y, test_X, test_y, reservoir_size=100, a=0.8, per_time_step=False, gridsearch=True, gridsearch_training_frac=0.7, error='accuracy'):
        # Inspired by http://minds.jacobs-university.de/mantas/code

        if gridsearch:
            reservoir_size, a = self.gridsearch_reservoir_computing(train_X, train_y, test_X, test_y, per_time_step=per_time_step, gridsearch_training_frac=gridsearch_training_frac, error=error)

        # We assume these parameters as fixed, but feel free to change them as well.
        washout_period = 10

        # Create a numerical dataset without categorical attributes.
        new_train_X, new_test_X = self.create_numerical_multiple_dataset(train_X, test_X)
        if test_y is None:
            new_train_y = self.create_numerical_single_dataset(train_y)
            new_test_y = None
        else:
            new_train_y, new_test_y = self.create_numerical_multiple_dataset(train_y, test_y)

        # We normalize the input.....
        new_train_X, new_test_X, min_X, max_X = self.normalize(new_train_X, new_test_X, 0, 1)
        new_train_y, new_test_y, min_y, max_y = self.normalize(new_train_y, new_test_y, -0.9, 0.9)

        inputs = len(new_train_X.columns)
        outputs = len(new_train_y.columns)

        # Randomly initialize our weight vectors.
        Win, W, Wback = self.initialize_echo_state_network(inputs, outputs, reservoir_size)


        # Allocate memory for our result matrices.
        X = np.zeros((len(train_X.index)-washout_period, 1+inputs+reservoir_size))
        Yt = new_train_y.iloc[washout_period:len(new_train_y.index),:].values
        Yt = np.arctanh( Yt )
        x = np.zeros((reservoir_size,1))

        # Train over all time points.
        for t in range(0, len(new_train_X.index)):

            # Set the inputs according to the values seen in the training set.
            u = new_train_X.iloc[t,:].values

            # Set the previous target value to the real value if available.
            if t > 0:
                y_prev= new_train_y.iloc[t-1,:].values
            else:
                y_prev = np.array([0]*outputs)

            # Determine the activation of the reservoir.
            x = (1-a)*x + a*np.tanh(np.dot(Win, np.vstack(np.insert(u,0,1)) ) + np.dot( W, x ) + np.dot( Wback, np.vstack(y_prev) ))

            # And store the values obtained after the washout period.
            if t >= washout_period:
                X[t-washout_period,:] = np.hstack(np.insert(np.insert(x, 0, u), 0, 1))


        # Train Wout.
        X_p = linalg.pinv(X)
        Wout = np.transpose(np.dot( X_p, Yt ))

        # And predict for both training and test set.


        pred_train_y, pred_train_y_prob = self.predict_values_echo_state_network(Win, W, Wback, Wout, a, reservoir_size, new_train_X, new_train_y, new_train_y.columns, per_time_step)
        pred_test_y, pred_test_y_prob = self.predict_values_echo_state_network(Win, W, Wback, Wout, a, reservoir_size, new_test_X, new_test_y, new_train_y.columns, per_time_step)

        pred_train_y_prob = self.denormalize(pred_train_y_prob, min_y, max_y, -0.9, 0.9)
        pred_test_y_prob = self.denormalize(pred_test_y_prob, min_y, max_y, -0.9, 0.9)

        return pred_train_y, pred_test_y, pred_train_y_prob, pred_test_y_prob

    # Creates a recurrent neural network dataset according to the pybrain specification.
    # Returns this new format.
    def rnn_dataset(self, X, y):

        # Create an empty dataset.
        ds = SequentialDataSet(len(X.columns), len(y.columns))

        # And add all rows...
        for i in range(0, len(X.index)):
            ds.addSample(tuple(X.iloc[i,:].values), tuple(y.iloc[i,:].values))
        return ds

    # Do a gridsearch for the recurrent neural network...
    def gridsearch_recurrent_neural_network(self, train_X, train_y, test_X, test_y, error='accuracy', gridsearch_training_frac=0.7):
        tuned_parameters = {'n_hidden_neurons': [50, 100], 'iterations':[250, 500], 'outputbias': [True]}
        params = list(tuned_parameters.keys())
        combinations = self.generate_parameter_combinations(tuned_parameters, params)
        split_point = int(gridsearch_training_frac * len(train_X.index))
        train_params_X = train_X.iloc[0:split_point,]
        test_params_X = train_X.iloc[split_point:len(train_X.index),]
        train_params_y = train_y.iloc[0:split_point,]
        test_params_y = train_y.iloc[split_point:len(train_X.index),]

        if error == 'mse':
            best_error = sys.float_info.max
        elif error == 'accuracy':
            best_error = 0

        best_combination = []
        for comb in combinations:
            print(comb)
            # Order of the keys might have changed.
            keys = list(tuned_parameters.keys())
            # print(keys)
            pred_train_y, pred_test_y, pred_train_y_prob, pred_test_y_prob = self.recurrent_neural_network(
                train_params_X, train_params_y, test_params_X, test_params_y,
                n_hidden_neurons=comb[keys.index('n_hidden_neurons')],
                iterations=comb[keys.index('iterations')],
                outputbias=comb[keys.index('outputbias')], gridsearch=False
            )

            if error == 'mse':
                eval = RegressionEvaluation()
                mse = eval.mean_squared_error(test_params_y, pred_test_y_prob)
                if mse < best_error:
                    best_error = mse
                    best_combination = comb
            elif error == 'accuracy':
                eval = ClassificationEvaluation()
                acc = eval.accuracy(test_params_y, pred_test_y)
                if acc > best_error:
                    best_error = acc
                    best_combination = comb
        print ('-------')
        print (best_combination)
        print ('-------')
        return best_combination[params.index('n_hidden_neurons')], best_combination[params.index('iterations')], best_combination[params.index('outputbias')]

    # Apply a recurrent neural network for classification upon the training data (with the specified number of
    # hidden neurons and iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def recurrent_neural_network(self, train_X, train_y, test_X, test_y, n_hidden_neurons=50, iterations=100, gridsearch=False, gridsearch_training_frac=0.7, outputbias=False, error='accuracy'):

        if gridsearch:
            n_hidden_neurons, iterations, outputbias = self.gridsearch_recurrent_neural_network(train_X, train_y, test_X, test_y, gridsearch_training_frac=gridsearch_training_frac, error=error)
        # Create numerical datasets first.
        new_train_X, new_test_X = self.create_numerical_multiple_dataset(train_X, test_X)
        new_train_y, new_test_y = self.create_numerical_multiple_dataset(train_y, test_y)

        # We normalize the input.....
        new_train_X, new_test_X, min_X, max_X = self.normalize(new_train_X, new_test_X, 0, 1)
        new_train_y, new_test_y, min_y, max_y = self.normalize(new_train_y, new_test_y, 0.1, 0.9)

        # Create the proper pybrain datasets.
        ds_training = self.rnn_dataset(new_train_X, new_train_y)
        ds_test = self.rnn_dataset(new_test_X, new_test_y)

        inputs = len(new_train_X.columns)
        outputs = len(new_train_y.columns)

        # Build the network with the proper parameters.
        n = buildNetwork(inputs, n_hidden_neurons, outputs, hiddenclass=SigmoidLayer, outclass=SigmoidLayer, outputbias=outputbias, recurrent=True)

        # Train using back propagation through time.
        #trainer = BackpropTrainer(n, dataset=ds_training, verbose=False, momentum=0.9, learningrate=0.01)
        trainer = RPropMinusTrainer(n, dataset=ds_training, verbose=False)

        for i in range(0, iterations):
            trainer.train()

#        for mod in n.modules:
#            for conn in n.connections[mod]:
#                print conn
#                for cc in range(len(conn.params)):
#                    print conn.whichBuffers(cc), conn.params[cc]

        # Determine performance on the training and test set....
#        Y_train = []
#        for i in range(0, len(new_train_X.index)):
#            input = tuple(new_train_X.iloc[i,:].values)
#            output = n.activate(input)
#            Y_train.append(output)
#        Y_test = []
#        for i in range(0, len(new_test_X.index)):
#            Y_test.append(n.activate(tuple(new_test_X.iloc[i,:].values)))

        Y_train = []
        Y_test = []

        for sample, target in ds_training.getSequenceIterator(0):
            Y_train.append(n.activate(sample).tolist())

        for sample, target in ds_test.getSequenceIterator(0):
            Y_test.append(n.activate(sample).tolist())

        y_train_result = pd.DataFrame(Y_train, columns=new_train_y.columns, index=train_y.index)
        y_test_result = pd.DataFrame(Y_test, columns=new_test_y.columns, index=test_y.index)

#        print y_train_result

        y_train_result = self.denormalize(y_train_result, min_y, max_y, 0.1, 0.9)
        y_test_result = self.denormalize(y_test_result, min_y, max_y, 0.1, 0.9)

#        plot.plot(train_y.index, train_y)
#        plot.hold(True)
#        plot.plot(train_y.index, pred_train_y_prob)
#        plot.show()


        return y_train_result.idxmax(axis=1), y_test_result.idxmax(axis=1), y_train_result, y_test_result

# The class includes several algorithm that capture the temporal dimension explicitly for
# classification problems.
class TemporalRegressionAlgorithms:

    # Applies a known dynamical systems model for a regression problem by tuning its parameters towards the data.
    # Hereto, it can use multiple objectives as it uses the nsga 2 algorithm. To be provided are:
    # training set (both input and target)
    # test set (both input and target)
    # a list of columns the model addresses (i.e. the states), the string should be preceded by 'self.' in order for the approach to work.
    # a list of equations to derive the specified states, again using 'self.' preceding all parameters and columns names.
    # a list of targets (a subset of the columns) (again with 'self.')
    # a list of parameters in the equations (again with 'self.')
    # the population size of nsga 2
    # the maximum number of generations for nsga 2
    # whether we want to predict per time point (i.e. we reset the state values of the previous time point to their
    # observed values.
    # It returns a series of predictions for the training and test sets that are the results of parameter setting that are positioned
    # on the Pareto front.
    def dynamical_systems_model_nsga_2(self, train_X, train_y, test_X, test_y, columns, equations, targets, parameters, pop_size=10, max_generations=100, per_time_step=True):
        prng = random.Random()
        evaluator = Evaluator()
        model = Model()

        # Create the model.
        model.set_model(columns, equations, parameters)

        # Set the desired/known values in our evaluator.
        evaluator.set_values(model, train_X, train_y, test_X, test_y, targets)

        # Initialize the NSGA2 algorithm.
        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [inspyred.ec.variators.blend_crossover,
                       inspyred.ec.variators.gaussian_mutation]
        ea.terminator = inspyred.ec.terminators.generation_termination

        # Let it run.
        final_pop = ea.evolve(generator=evaluator.generator, evaluator=evaluator.evaluator_multi_objective, pop_size=pop_size, maximize=False, bounder=None,max_generations=max_generations)
        final_arc = ea.archive

        # For all solutions (they reside on the pareto front)
        return_values = []
        for f in final_arc:

            # Predict the results.
            train_fitness, y_train_pred = evaluator.predict(f.candidate, training=True, per_time_step=per_time_step)
            test_fitness, y_test_pred = evaluator.predict(f.candidate, training=False, per_time_step=per_time_step)

            # And collect the predictions and fitness values.
            row = [y_train_pred, train_fitness, y_test_pred, test_fitness]
            return_values.append(row)
        return return_values

    # Applies a known dynamical systems model for a regression problem by tuning its parameters towards the data using a GA.
    # Hereto, it can use multiple objectives but will just average the values of each one. In the end, one solution
    # will be provided. To be provided are:
    # training set (both input and target)
    # test set (both input and target)
    # a list of columns the model addresses (i.e. the states), the string should be preceded by 'self.' in order for the approach to work.
    # a list of equations to derive the specified states, again using 'self.' preceding all parameters and columns names.
    # a list of targets (a subset of the columns) (again with 'self.')
    # a list of parameters in the equations (again with 'self.')
    # the population size of the GA
    # the maximum number of generations for the GA.
    # whether we want to predict per time point (i.e. we reset the state values of the previous time point to their
    # observed values.
    # It returns a prediction for the training and test sets that are the result of the best parameter setting.
    def dynamical_systems_model_ga(self, train_X, train_y, test_X, test_y, columns, equations, targets, parameters, pop_size=10, max_generations=100, per_time_step=True):
        prng = random.Random()
        evaluator = Evaluator()
        model = Model()

        # Create the model.
        model.set_model(columns, equations, parameters)

        # Set the desired/known values in our evaluator.
        evaluator.set_values(model, train_X, train_y, test_X, test_y, targets)
        ea = inspyred.ec.GA(prng)
        ea.terminator = inspyred.ec.terminators.generation_termination

        # Let it run.
        final_pop = ea.evolve(generator=evaluator.generator, evaluator=evaluator.evaluator_single_objective, pop_size=pop_size, maximize=False, bounder=None,max_generations=max_generations)

        # Select the best one and use it to predict.
        best = min(final_pop)
        train_fitness, y_train_pred = evaluator.predict(best.candidate, training=True, per_time_step=per_time_step)
        test_fitness, y_test_pred = evaluator.predict(best.candidate, training=False, per_time_step=per_time_step)
        return y_train_pred, y_test_pred

    # Applies a known dynamical systems model for a regression problem by tuning its parameters towards the data using SA.
    # Hereto, it can use multiple objectives but will just average the values of each one. In the end, one solution
    # will be provided. To be provided are:
    # training set (both input and target)
    # test set (both input and target)
    # a list of columns the model addresses (i.e. the states), the string should be preceded by 'self.' in order for the approach to work.
    # a list of equations to derive the specified states, again using 'self.' preceding all parameters and columns names.
    # a list of targets (a subset of the columns) (again with 'self.')
    # a list of parameters in the equations (again with 'self.')
    # the population size for SA
    # the maximum number of generations for the SA.
    # whether we want to predict per time point (i.e. we reset the state values of the previous time point to their
    # observed values.
    # It returns a prediction for the training and test sets that are the result of the best parameter setting.
    def dynamical_systems_model_sa(self, train_X, train_y, test_X, test_y, columns, equations, targets, parameters, pop_size=10, max_generations=100, per_time_step=True):
        prng = random.Random()
        evaluator = Evaluator()
        model = Model()

        # Create the model.
        model.set_model(columns, equations, parameters)

        # Set the desired/known values in our evaluator.
        evaluator.set_values(model, train_X, train_y, test_X, test_y, targets)
        ea = inspyred.ec.SA(prng)
        ea.terminator = inspyred.ec.terminators.generation_termination

        # Let it run.
        final_pop = ea.evolve(generator=evaluator.generator, evaluator=evaluator.evaluator_single_objective, maximize=False, bounder=None, max_generations=max_generations)

        # Select the best one and use it to predict.
        best = min(final_pop)
        train_fitness, y_train_pred = evaluator.predict(best.candidate, training=True, per_time_step=per_time_step)
        test_fitness, y_test_pred = evaluator.predict(best.candidate, training=False, per_time_step=per_time_step)
        return y_train_pred, y_test_pred

    # Apply an echo state network for regression upon the training data (with the specified reservoir size),
    # and use the created network to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    def reservoir_computing(self, train_X, train_y, test_X, test_y, reservoir_size=100, a=0.8, per_time_step=False, gridsearch=True, gridsearch_training_frac=0.7):
        # Simply apply the classification variant, but only consider the numerical predictions.
        tc = TemporalClassificationAlgorithms()
        pred_train_y, pred_test_y, pred_train_y_val, pred_test_y_val = tc.reservoir_computing(train_X, train_y, test_X, test_y, reservoir_size=reservoir_size, a=a, per_time_step=per_time_step, gridsearch=gridsearch, gridsearch_training_frac=gridsearch_training_frac, error='mse')
        return pred_train_y_val, pred_test_y_val


    # Apply a recurrent neural network for regression upon the training data (with the specified number of
    # hidden neurons and iterations), and use the created network to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set.
    def recurrent_neural_network(self, train_X, train_y, test_X, test_y, n_hidden_neurons=50, iterations=100, gridsearch=False, gridsearch_training_frac=0.7, outputbias=False):
        # Simply apply the classification variant, but only consider the numerical predictions.
        tc = TemporalClassificationAlgorithms()
        pred_train_y, pred_test_y, pred_train_y_val, pred_test_y_val = tc.recurrent_neural_network(train_X, train_y, test_X, test_y, n_hidden_neurons=n_hidden_neurons, iterations=iterations, gridsearch=gridsearch, gridsearch_training_frac=gridsearch_training_frac, outputbias=outputbias, error='mse')
        return pred_train_y_val, pred_test_y_val

    # Do a gridsearch for the time series.
    def gridsearch_time_series(self, train_X, train_y, test_X, test_y, error = 'mse', gridsearch_training_frac=0.7):
        tuned_parameters = {'ar': [0, 5], 'ma':[0, 5], 'd':[1]}
        params = list(tuned_parameters.keys())

        tc = TemporalClassificationAlgorithms()
        combinations = tc.generate_parameter_combinations(tuned_parameters, params)
        split_point = int(gridsearch_training_frac * len(train_X.index))
        train_params_X = train_X.iloc[0:split_point,]
        test_params_X = train_X.iloc[split_point:len(train_X.index),]
        train_params_y = train_y.iloc[0:split_point,]
        test_params_y = train_y.iloc[split_point:len(train_X.index),]

        if error == 'mse':
            best_error = sys.float_info.max
        elif error == 'accuracy':
            best_error = 0

        best_combination = []
        for comb in combinations:
            print(comb)
            # Order of the keys might have changed.
            keys = list(tuned_parameters.keys())
            pred_train_y, pred_test_y = self.time_series(train_params_X, train_params_y, test_params_X, test_params_y,
                                                                                                     ar=comb[keys.index('ar')], ma=comb[keys.index('ma')],d=comb[keys.index('d')],
                                                                                                     gridsearch=False)

            eval = RegressionEvaluation()
            mse = eval.mean_squared_error(test_params_y, pred_test_y)
            if mse < best_error:
                best_error = mse
                best_combination = comb

        print('-------')
        print(best_combination)
        print('-------')
        return best_combination[keys.index('ar')], best_combination[keys.index('ma')], best_combination[keys.index('d')]


    # Apply a time series ARIMAX approach, and use the created network to predict the outcome for both the
    # test and training set. It returns the predictions for the training and test set. Parameters can be
    # provided around the learning algorithm and a grid search can also be performed.
    def time_series(self, train_X, train_y, test_X, test_y, ar=1, ma=1, d=0, gridsearch=False, gridsearch_training_frac=0.7):
        if gridsearch:
            ar, ma, d = self.gridsearch_time_series(train_X, train_y, test_X, test_y, gridsearch_training_frac=gridsearch_training_frac, error='mse')

        train_dataset = copy.deepcopy(train_X)
        formula = train_y.name + '~1+' + "+".join(train_X.columns)
        train_dataset[train_y.name] = train_y
        test_dataset = copy.deepcopy(test_X)
        test_dataset[test_y.name] = test_y

        model = pf.ARIMAX(data=train_dataset,formula=formula,ar=ar,ma=ma)
        x = model.fit()
        #x.summary()
        model_pred = model.predict(h=len(train_y.index)-max(ar, ma), oos_data=train_dataset)
        values = np.empty((len(model_pred) + max(ar, ma),1))
        values[:] = np.nan
        values[max(ar, ma):] = model_pred.values
        pred_train = pd.DataFrame(values, index=train_y.index, columns=[train_y.name])
        pred_train.iloc[max(ar, ma):,:] = model_pred.values
        pred_test = pd.DataFrame(model.predict(h=len(test_y.index), oos_data=test_dataset).values, index=test_y.index, columns=[test_y.name])

        return pred_train, pred_test
