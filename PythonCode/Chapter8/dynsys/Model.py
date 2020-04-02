##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 8                                               #
#                                                            #
##############################################################

import math

# The class represents a dynamical systems model.
class Model:

    state_names = []
    state_values = []
    predicted_values = []
    state_equations = []
    parameter_names = []
    parameter_values = []
    t = 0
    max_value = 10000

    def __init__(self):
        self.state_names = []
        self.state_values = []
        self.predicted_values = []
        self.state_equations = []
        self.parameter_names = []
        self.parameter_values = []
        self.t = 0

    # This function sets the model in the global variables. It uses the state names, equations
    # in the same order, and the parameter names.
    def set_model(self, state_names, state_equations, parameter_names):
        self.state_names = state_names
        self.state_values.append([])
        self.state_equations = state_equations
        self.parameter_names = parameter_names

    # This function resets the model, it empties the predictions for the states and sets
    # the time point to 0.
    def reset(self):
        self.t = 0
        self.state_values = []
        self.state_values.append([])
        self.predicted_values = []
        self.predicted_values.append([])

    # This function sets the parameter values in the model.
    def set_parameter_values(self, param_values):
        for p in range(len(self.parameter_names)):
            # We do a bit of magic here, since we do not know the variable
            # names for the parameters up front we execute it in this
            # way. This results in global variables with the proper values.
            exec("%s = %f" % (self.parameter_names[p], param_values[p]))
            self.parameter_values.append(param_values[p])

    # This functions sets the state values in the model.
    def set_state_values(self, state_values):
        for s in range(len(self.state_names)):
            # We do a bit of magic here, since we do not know the variable
            # names for the states up front we execute it in this
            # way. This results in global variables with the proper values.
            exec("%s = %f" % (self.state_names[s], state_values[s]))
            self.state_values[self.t].append(state_values[s])

    # Some basic printing of the model.
    def print_model(self):
        for e in range(len(self.state_equations)):
            print(str(self.state_names[e]) + ' = ',)
            print(self.state_equations[e])

    # Prints the model to a file with the generation of the high level optimization algorithm.
    def print_model_to_file(self, file, generation):
        file.write('======================' + str(generation) + '======================\n')
        for e in range(len(self.state_equations)):
            file.write(str(self.state_names[e]) + ' = ' + str(self.state_equations[e]) + '\n')

    # Return the model in a string representation.
    def to_string(self):
        result = ''
        for e in range(len(self.state_equations)):
            result += str(self.state_equations[e])
        return result

    # Executes the model for the given number of time steps, given the current
    # settings for the states.
    def execute_steps(self, steps):

        # Repeat for the given number of time steps.
        for i in range(0,steps):

            # Allocate memory for the state values and the predicted values.
            self.state_values.append([0]*len(self.state_names))
            self.predicted_values.append([0]*len(self.state_names))
            self.t += 1

            # Compute the predicted values based on the current values for the states.
            for v in range(len(self.state_names)):

                # We compute the value of the state equation.
                value = eval(self.state_equations[v])

                # If the number is fishy, we select the maximum value.
                if math.isinf(value) or math.isnan(value):
                    value = self.max_value

                # And we set the value of the state accordingly.
                exec("%s = %f" % (self.state_names[v], value))

                # For debugging.
                self.state_values[self.t][v] = eval(self.state_names[v])

                # And we add the prediction for the time point to the file.
                self.predicted_values[self.t][v] = self.state_values[self.t][v]


    # This function return the values of the specified state from time point 0 to now.
    def get_values(self, state):

        # If we do not have any values, we do not do anything.
        if self.t == 0:
            print('number of values ' + str(self.t))
        values = []

        # Get the index of the state
        index = self.state_names.index(state)

        # And get the values over all time points.
        for i in range(1, len(self.predicted_values)):
            value = self.predicted_values[i][index]
            values.append(value)
        return values