__author__ = 'misha'

import numpy as np


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


class BaseLayer(object):
    """
    Base layer for hidden and output layers with logistic activation function
    """

    def __init__(self, size, input_size, theta=None):
        """
        Initialize with size and the size of the previous layer.
        Activations of the previous layer are inputs for this layer
        :param size: Number of neurons in this layer
        :param input_size: Size of the previous layer
        :param theta: optionally set starting theta
        """
        self.size = size
        self.activations = None

        self.input_size = input_size
        self.inputs = None
        self.input_plus_bias = None
        self.deltas = None

        # start with random weights
        if theta is None:
            self.theta = 0.1 * np.random.randn(input_size + 1, size)
        else:
            self.theta = theta

        # these are derivatives of cost function with respect to weights, used for updating the weights during training
        self.theta_update = np.zeros(self.theta.shape)

        # number of training examples seen since last update of the weights
        self.batch_counter = 0

    @staticmethod
    def activation_function(arr):
        """
        Logistic function
        :param arr: input array of arguments
        :return: array of results
        """
        arr = np.array(arr)
        return 1. / (1. + np.exp(- arr))

    @staticmethod
    def activation_derivative(arr):
        """
        Derivative of activation function
        :return: derivative values
        """
        return arr * (1. - arr)

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
        self.activations = self.activation_function(z)

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
            self.deltas = self.activation_derivative(self.activations) * pre_d

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


class LogisticLayer(BaseLayer):
    """Just for naming"""
    pass


class LogisticTLayer(BaseLayer):
    """
    Same as Base layer, but with "temperature" in the activation function
    """
    def __init__(self, size, input_size, theta=None, T=1.):

        super(LogisticTLayer, self).__init__(size, input_size, theta)

        self.T = T

    def activation_function(self, arr):
        """
        Logistic function
        :param arr: input array of arguments
        :return: array of results
        """
        arr = np.array(arr)
        return 1. / (1. + np.exp(- arr / self.T))

    def activation_derivative(self, arr):
        """
        Derivative of activation function
        :return: derivative values
        """
        return arr * (1. - arr) / self.T



class RectifierLayer(BaseLayer):
    """
    Same as Base layer, but with rectifier activation function
    """
    def __init__(self, size, input_size, theta=None, threshold=0.):

        super(RectifierLayer, self).__init__(size, input_size, theta)

        self.threshold = threshold


    def activation_function(self, arr):
        """
        Linear rectifier (hard max) activation function
        :param arr:
        :return:
        """
        arr = np.subtract(arr, self.threshold)
        return (arr > 0) * arr

    def activation_derivative(self, arr):
        """
        Derivative of rectifier
        :param arr:
        :return:
        """
        return (np.subtract(arr, self.threshold) > 0) * 1.


class RectifierInhibitLayer(BaseLayer):
    """
    Same as Base layer, but with rectifier activation function
    """
    def __init__(self, *args, **kwargs):
        super(RectifierInhibitLayer, self).__init__(*args, **kwargs)

        #self.inhibitors = (np.random.randn(self.size) > 0) * 2. - 1.

    def activation_function(self, arr):
        """
        Linear rectifier (hard max) activation function
        :param x:
        :return:
        """
        out = (arr > 0) * arr
        out[:, : self.size / 2] += -1.
        return out

    def activation_derivative(self, arr):
        """
        Derivative of rectifier
        :param arr:
        :return:
        """
        out = (arr > 0) * 1.
        out[:, : self.size / 2] += -1.
        return out

