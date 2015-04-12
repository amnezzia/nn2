__author__ = 'misha'

import numpy as np


class UnknownActivationFunction(Exception):
    """Exception when specified activation string is not one of the implemented"""
    pass


class InputLayer(object):
    """
    Contains the basics: activations and size
    Will serve as a base class
    """

    def __init__(self, size):

        self.size = size
        self.activations = None

    def activate(self, inputs):
        """
        In this case activations are inputs
        """

        self.activations = np.array(inputs).reshape((-1, self.size))


class HiddenLayer(object):
    """
    Hidden layer
    """

    def __init__(self, size, input_size, activate_f='logit', theta=None):
        """
        Initialize with size and the size of the previous layer.
        Activations of the previous layer are inputs for this layer
        :param size: Number of neurons in this layer
        :param input_size: Size of the previous layer
        :param activate_f: Activation function, default: 'log', right now only logistic function is implemented
        :param theta: optionally set starting theta
        """
        self.size = size
        self.activations = None

        # for being able to specify the activation function
        self.activate_f = activate_f

        self.input_size = input_size
        self.inputs = None
        self.input_plus_bias = None
        self.deltas = None

        # start with random weights
        if theta is None:
            self.theta = 0.1 * np.random.rand(input_size + 1, size)
        else:
            self.theta = theta
        # these are derivatives of cost function with respect to weights, used for updating the weights during training
        self.theta_update = np.zeros(self.theta.shape)

        # number of training examples seen since last update of the weights
        self.batch_counter = 0

    @staticmethod
    def _logistic(arr):
        """
        Logistic function
        :param arr: input array of arguments
        :return: array of results
        """
        arr = np.array(arr)
        return 1. / (1. + np.exp(- arr))

    def _activation_function(self, x):
        """
        This is for being able to switch between different activation functions
        :param x: array of input arguments
        :return: results
        """
        if self.activate_f == 'logit':
            return self._logistic(x)
        elif self.activate_f == 'lrect':
            return self._linear_rectifier(x)
        else:
            raise UnknownActivationFunction("{} is unknown activation function".format(self.activate_f))

    def _activation_derivative(self,):
        """
        Selection of activation function derivatives
        :return: derivative values
        """
        if self.activations is None:
            return None
        elif self.activate_f == 'logit':
            return self.activations * (1. - self.activations)
        #elif self.activate_f == 'lrect':
        #    return self.activations * (1. - self.activations)
        else:
            raise UnknownActivationFunction("{} is unknown activation function".format(self.activate_f))

    def _set_inputs(self, inputs):
        """
        Helper method to set layer inputs, make sure the shape is correct and to add biases
        :param inputs: inputs to the layer (activations of the previous layer)
        """
        self.inputs = np.array(inputs).reshape((-1, self.input_size))

        bias = np.ones((self.inputs.shape[0], 1))
        self.input_plus_bias = np.concatenate((bias, self.inputs), axis=1)

    def activate(self, inputs):
        """
        Propagates forward: calculates activations of this layer neurons from given inputs
        :param inputs: layer inputs (activations of the previous layer)
        :return:
        """
        self._set_inputs(inputs)

        # sum up all inputs times the weights
        z = np.dot(self.input_plus_bias, self.theta)
        # calculate activations using activation function
        self.activations = self._activation_function(z)

    def set_deltas(self, pre_deltas):
        """
        Calculates deltas from propagated back deltas (or form targets)
        :param pre_deltas: deltas x weights of the previous layer (or difference between outputs and targets)
        :return:
        """
        if self.activations is None:
            pass
        else:
            pre_d = np.array(pre_deltas).reshape((-1, self.size))
            self.deltas = self._activation_derivative() * pre_d

    def get_previous_pre_deltas(self,):
        """
        Calculate deltas x weights for the lower level
        :return:
        """
        # remove the biases from result
        return  np.dot(self.deltas, self.theta.T)[:, 1:]

    def increment_updates(self,):
        """
        Add to the cost derivatives accumulator using deltas
        """
        self.theta_update += np.dot(self.input_plus_bias.T, self.deltas)
        # increment the number of seen examples
        self.batch_counter += self.deltas.shape[0]

    def update_weights(self, alpha, la):
        """
        Update weights
        :param alpha: learning rate
        :param la: lambda - regularization coefficient
        """
        # update
        self.theta -= float(alpha) * (self.theta_update / self.batch_counter + la * self.theta)
        # reset examples counter and derivatives accumulator
        self.theta_update = np.zeros(self.theta.shape)
        self.batch_counter = 0


class OutputLayer(HiddenLayer):
    """
    Same as Hidden layer, but with different deltas calculation
    """
    pass
    #def set_deltas(self, pre_deltas):
    #    """
    #    Deltas are just the difference between outputs and targets
    #    :param targets: Targets
    #    :return:
    #    """
    #    if self.activations is None:
    #        pass
    #    else:
    #        self.deltas = np.array(pre_deltas).reshape((-1, self.size)).astype(np.float)
