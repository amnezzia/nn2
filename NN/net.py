__author__ = 'misha'

import numpy as np

from layers import InputLayer, HiddenLayer, OutputLayer


class SamplesNumberMismatch(Exception):
    """Exception when number of training input samples does not match the number of target samples"""
    pass


class UnknownCostFunction(Exception):
    """Exception when specified cost function is not one of the implemented"""
    pass


class Net(object):
    """
    Class to assemble the network, train and use it
    """

    def __init__(self, layer_sizes, cost_f='log'):
        """
        Initialize network with the following parameters:
        :param layer_sizes: list of layer sizes including input and output([2, 3, 4] - 2 neurons in the input layer,
                            3 in the hidden layer, 4 in the output)
        :param cost_f: optional, default is 'log' for log cost function
        :return:
        """

        self.layer_sizes = layer_sizes
        self.cost_f = cost_f

        # list to keep track of cost values during training
        self.costs = []

        # outputs produced by the network, same as last layer activations
        self.outputs = None

        # instantiate layers
        self.layers = []
        for i, size in enumerate(self.layer_sizes):
            if i == 0:
                self.layers.append(InputLayer(size))
            elif i == len(self.layer_sizes) - 1:
                self.layers.append(OutputLayer(size, self.layer_sizes[i - 1]))
            else:
                self.layers.append(HiddenLayer(size, self.layer_sizes[i - 1]))

    def _log_cost(self, outputs, targets, reg_coeff):
        """
        Log cost function
        :param outputs: predicted outputs
        :param targets: targets
        :return: cost value
        """
        j = targets * np.log(outputs) + (1. - targets) * np.log(1. - outputs)
        thetas_sq_sum = sum([(l.theta ** 2).sum() for l in self.layers[1:]])
        e = (- j.sum() + reg_coeff * 0.5 * thetas_sq_sum) / float(j.shape[0])

        return e

    @staticmethod
    def _split_into_batches(arr, batch_size):
        """
        Helper method to split arrays into batches (used for splitting training input and target arrays)
        :param arr: given array of data
        :param batch_size: Directly specify batch size by passing an integer.
                        Alternatively can specify a fraction of the total length.
                        If 0, negative number or None is passed - will create just one batch with the whole array.
        :return: list of batches
        """
        arr_length = len(arr)
        batches = list()

        if batch_size is None or batch_size <= 0:
            batches.append(arr)
            return batches

        # convert float size into integers
        if type(batch_size) == float and 0 < batch_size < 1:
            batch_size = max(np.round(batch_size * arr_length), 1)

        # make sure batch_size is an integer (this will also deal with floats greater than 1)
        batch_size = np.int(batch_size)

        curr_index = 0
        while curr_index < arr_length:
            batches.append(arr[curr_index: curr_index + batch_size])
            curr_index += batch_size

        return batches

    def forward(self, batch_inputs):
        """
        Perform forward pass through the network
        :param batch_inputs: inputs to the network, can be single example or an array of examples
        """
        for i, l in enumerate(self.layers):
            if i == 0:
                l.activate(batch_inputs)
            else:
                l.activate(self.layers[i - 1].activations)

        # for convenience
        self.outputs = self.layers[-1].activations

    def backward(self, batch_targets):
        """
        Propagate back the deltas (errors)
        :param batch_targets: expected outputs of the network
        """
        # for the output layer pre-deltas are just targets, see OutputLayer class
        pre_d = batch_targets
        for l in self.layers[:0:-1]:
            # calculate deltas for the current layer
            l.set_deltas(pre_d)
            # calculate theta updates using these deltas
            l.increment_updates()
            # calculate pre-deltas for the previous layer
            pre_d = l.get_previous_pre_deltas()

    def update_weights(self, update_rate, reg_coeff):
        """
        Update weight for all layers
        :param update_rate: learning rate
        """
        for l in self.layers[:0:-1]:
            l.update_weights(update_rate, reg_coeff)

    def cost(self, targets, reg_coeff):
        """
        Calculate cost value for the current batch.
        :param targets: current batch expected targets
        :return:
        """
        if self.cost_f == 'log':
            return self._log_cost(self.outputs, targets, reg_coeff)
        else:
            raise UnknownCostFunction("{} - unknown cost function".format(self.cost_f))

    def fit(self, X, Y,
            batch_size=None,
            update_at='batch',
            reg_coeff=0.,
            update_rate=1.,
            max_iter=1000,
            min_cost=0.001):
        """
        Training of the network
        :param X: training inputs (list/array of examples)
        :param Y: training targets (list/array of examples)
        :param batch_size: size of batches to split training set into
        :param update_at: Specify when to update weights:
                        'batch' - after each batch,
                        'epoch' - after going through the whole dataset
                        If any other value, the weights will never be updated.
        :param reg_coeff: regularization coefficient
        :param update_rate: learning rate
        :param max_iter: Maximum number of epochs (going once through the whole dataset)
        :param min_cost: Stop training if cost below this value. The training will also stop if cost variation during
                        last 10 epochs was less than 0.01 * min_cost.
        """

        if len(Y) != len(X):
            raise SamplesNumberMismatch("Lengths of X and Y should be the same")

        # split into batches
        batches = zip(self._split_into_batches(X, batch_size),
                      self._split_into_batches(Y, batch_size))

        counter = 0
        stopping_condition = False

        while not stopping_condition and counter < max_iter:
            # cost accumulator
            j = 0
            for batch_inputs, batch_targets in batches:
                # forward
                self.forward(batch_inputs)

                # calculate cost and add to the accumulator
                j += self.cost(batch_targets, reg_coeff)

                # backward
                self.backward(batch_targets)

                # maybe update weights
                if update_at == 'batch':
                    self.update_weights(update_rate, reg_coeff)

            # maybe update weights
            if update_at == 'epoch':
                self.update_weights(update_rate, reg_coeff)

            # keep track of the cost values
            self.costs.append(j / float(len(batches)))

            counter += 1

            # update stopping condition
            stopping_condition = (self.costs[-1] < min_cost)
            if counter > 10:
                stopping_condition = stopping_condition or \
                                     (max(self.costs[-1:-10:-1]) - min(self.costs[-1:-10:-1])) < min_cost / 100.

    def predict(self, X):
        """
        Perform forward pass only and return the results
        :param X: list/array of inputs
        :return: list/array of resulting outputs
        """

        self.forward(X)
        return self.outputs


