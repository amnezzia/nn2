__author__ = 'misha'

import numpy as np

from layers import InputLayer, LogisticLayer, LogisticTLayer, RectifierLayer, RectifierInhibitLayer


class ArrayLengthMismatch(Exception):
    """Exception when number of training input samples does not match the number of target samples"""
    pass


class UnknownCostFunction(Exception):
    """Exception when specified cost function is not one of the implemented"""
    pass


class UnknownLayerType(Exception):
    """Exception when specified activation string is not one of the implemented"""
    pass


class Net(object):
    """
    Class to assemble the network, train and use it
    """

    def __init__(self, layer_sizes, types=None, cost_f='log', thetas=None):
        """
        Initialize network with the following parameters:
        :param layer_sizes: list of layer sizes including input and output([2, 3, 4] - 2 neurons in the input layer,
                            3 in the hidden layer, 4 in the output)
        :param types: optional list of layer types, list length should be one less the the length of sizes
                (if length of the list is the same as sizes, then first corresponds to the input layer and is ignored)
        :param cost_f: optional, default is 'log' for log cost function
        :param thetas: optional list of initial weights for each layer,
                list length should be one less the the length of sizes
                (if length of the list is the same as sizes, then first corresponds to the input layer and is ignored)
        :return:
        """

        self.layer_sizes = layer_sizes
        self.cost_f = cost_f

        # list to keep track of cost values during training
        self.costs = []
        self.costs_batches = []

        # assign cost function
        self._assign_cost_function()

        # outputs produced by the network, same as last layer activations
        self.outputs = None

        # prepare thetas
        layer_thetas = self._prepare_parameters_list(thetas)

        # prepare types
        layer_types = self._prepare_parameters_list(types)

        # instantiate layers
        self.layers = []
        for i, size in enumerate(self.layer_sizes):
            if i == 0:
                self.layers.append(InputLayer(size))
            else:
                new_layer = self._get_new_layer(layer_types[i],
                                                size,
                                                self.layer_sizes[i - 1],
                                                theta=layer_thetas[i])
                self.layers.append(new_layer)

    def _prepare_parameters_list(self, in_list):
        """
        Helper method to standardize various parameters for layers instantiations
        :param in_list: what is passed to __init__
        :return: list of parameters with length equal to the total number of layers
        """
        out_list = [None for _ in self.layer_sizes]
        if in_list is None:
            pass
        elif len(in_list) == len(self.layer_sizes):
            out_list = in_list
        elif len(in_list) == len(self.layer_sizes) - 1:
            out_list[1:] = in_list
        else:
            raise ArrayLengthMismatch("Number of layer parameters should be equal "
                                      "to or one less than the number of layers")

        return out_list

    @staticmethod
    def _get_new_layer(layer_type, *args, **kwargs):
        """
        Method to instantiate a proper layer type
        :param type: type of the layer. If None - returns standard BaseLayer with logistic activation function.
        :param args: arguments for layer instantiation
        :param kwargs: keyword arguments for layer instantiation
        :return: new layer
        """
        if layer_type is None or layer_type == 'logit':
            new_layer = LogisticLayer(*args, **kwargs)
        elif layer_type == 'logit_T':
            new_layer = LogisticTLayer(*args, **kwargs)
        elif layer_type == 'lrect':
            new_layer = RectifierLayer(*args, **kwargs)
        elif layer_type == 'lrect_i':
            new_layer = RectifierInhibitLayer(*args, **kwargs)
        else:
            raise UnknownLayerType("{} is not a valid layer type".format(layer_type))

        return new_layer

    def _assign_cost_function(self):
        """
        Helper method to set cost function
        :return:
        """
        self.cost = None
        if self.cost_f == 'log':
            self.cost = self._log_cost
            self.cost_derivative = self._log_cost_derivative
        elif self.cost_f == 'square':
            self.cost = self._square_cost
            self.cost_derivative = self._square_cost_derivative
        else:
            raise UnknownCostFunction("{} - unknown cost function".format(self.cost_f))

    def _log_cost(self, outputs, targets, reg_coeff):
        """
        Log cost function
        :param outputs: predicted outputs
        :param targets: targets
        :return: cost value
        """
        j = targets * np.log(outputs) + (1. - targets) * np.log(1. - outputs)
        cost = (- j.sum() + self._l2_regularization(reg_coeff)) / float(j.shape[0])

        return cost

    def _square_cost(self, outputs, targets, reg_coeff):
        """
        Square difference cost function
        :param outputs: predicted outputs
        :param targets: targets
        :return: cost value
        """
        j = 0.5 * (outputs - targets) ** 2
        cost = (j.sum() + self._l2_regularization(reg_coeff)) / float(j.shape[0])

        return cost

    def _l2_regularization(self, reg_coeff):
        """
        Helper method to calculate regularization term for cost function
        :param reg_coeff:
        :return:
        """
        return 0.5 * reg_coeff * sum([(l.theta ** 2).sum() for l in self.layers[1:]])

    @staticmethod
    def _square_cost_derivative(outputs, targets):
        """
        Method to calculate pre-deltas for output layer
        (which is a derivative of square-loss cost function with respect to outputs)
        :return:
        """
        return np.subtract(outputs, targets)

    @staticmethod
    def _log_cost_derivative(outputs, targets):
        """
        Method to calculate pre-deltas for output layer
        (which is a derivative of log-loss cost function with respect to outputs)
        :return:
        """
        return np.subtract(outputs, targets) / (outputs * np.subtract(1., outputs))

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
        # for the output layer:
        pre_d = self.cost_derivative(self.layers[-1].activations, batch_targets)

        # finish for output and continue for all others
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

        if type(update_rate) == float or type(update_rate) == int:
            for l in self.layers[:0:-1]:
                l.update_weights(update_rate, reg_coeff)
        elif len(self.layers) == len(update_rate):
            for ur, l in zip(update_rate, self.layers)[:0:-1]:
                l.update_weights(ur, reg_coeff)
        else:
            raise Exception("Bad update rate parameters")

    def fit(self, X, Y,
            batch_size=None,
            reg_coeff=0.,
            update_rate=1.,
            max_iter=50,
            min_cost=0.001,
            cross_val_X=None,
            cross_val_Y=None):
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
            raise ArrayLengthMismatch("Lengths of X and Y should be the same")

        # split into batches
        batches = zip(self._split_into_batches(X, batch_size),
                      self._split_into_batches(Y, batch_size))

        # use cross validation
        use_cross_val = False
        if cross_val_X is not None and cross_val_Y is not None:
            cross_val_X = np.array(cross_val_X).reshape((-1, X.shape[1]))
            cross_val_Y = np.array(cross_val_Y).reshape((-1, Y.shape[1]))

            if len(cross_val_X) == len(cross_val_Y):
                use_cross_val = True
            else:
                print "Lengths of cross validation inputs and outputs do not match. Will not use."

        # start the loop
        counter = 0
        stopping_condition = False

        while not stopping_condition and counter < max_iter:
            # cost accumulator
            j = 0
            for batch_inputs, batch_targets in batches:

                # forward
                self.forward(batch_inputs)

                # calculate cost and add to the accumulator
                if not use_cross_val:
                    c = self.cost(self.outputs, batch_targets, reg_coeff)
                    self.costs_batches.append(c)
                    j += c

                # backward
                self.backward(batch_targets)

                # update weights
                self.update_weights(update_rate, reg_coeff)

            # keep track of the cost values
            if use_cross_val:
                self.forward(cross_val_X)
                j = self.cost(self.outputs, cross_val_Y, reg_coeff)
                self.costs.append(j)
            else:
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


class StackedAutoEncoders(Net):

    def fit(self, X, Y,
            batch_size=None,
            reg_coeff=0.,
            update_rate=1.,
            max_iter=50,
            min_cost=0.001,
            cross_val_X=None,
            cross_val_Y=None):

        # train first layer
        # forward to the first layer
        # train second
        # forward to the second...
        pass

class Net2(Net):

    @staticmethod
    def _split_into_batches(arr_length, batch_size):
        """
        Helper method to split arrays into batches (used for splitting training input and target arrays)
        :param arr: given array of data
        :param batch_size: Directly specify batch size by passing an integer.
                        Alternatively can specify a fraction of the total length.
                        If 0, negative number or None is passed - will create just one batch with the whole array.
        :return: list of batches
        """
        batches = list()

        if batch_size is None or batch_size <= 0:
            batches.append((0, -1))
            return batches

        # convert float size into integers
        if type(batch_size) == float and 0 < batch_size < 1:
            batch_size = max(np.round(batch_size * arr_length), 1)

        # make sure batch_size is an integer (this will also deal with floats greater than 1)
        batch_size = np.int(batch_size)

        curr_index = 0
        while curr_index < arr_length:
            # just save indexes instead of full array
            batches.append((curr_index, curr_index + batch_size))
            curr_index += batch_size

        return batches


    def fit(self, X, Y,
            batch_size=None,
            reg_coeff=0.,
            update_rate=1.,
            max_iter=50,
            min_cost=0.001,
            cross_val_X=None,
            cross_val_Y=None):
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
            raise ArrayLengthMismatch("Lengths of X and Y should be the same")

        # split into batches
        batches = self._split_into_batches(len(X), batch_size)

        # use cross validation
        use_cross_val = False
        if cross_val_X is not None and cross_val_Y is not None:
            cross_val_X = np.array(cross_val_X).reshape((-1, X.shape[1]))
            cross_val_Y = np.array(cross_val_Y).reshape((-1, Y.shape[1]))

            if len(cross_val_X) == len(cross_val_Y):
                use_cross_val = True
            else:
                print "Lengths of cross validation inputs and outputs do not match. Will not use."

        # start the loop
        counter = 0
        stopping_condition = False

        while not stopping_condition and counter < max_iter:
            # cost accumulator
            j = 0
            for start_index, end_index in batches:

                batch_inputs = X[start_index: end_index]
                batch_targets = Y[start_index: end_index]
                # forward
                self.forward(batch_inputs)

                # calculate cost and add to the accumulator
                if not use_cross_val:
                    c = self.cost(self.outputs, batch_targets, reg_coeff)
                    self.costs_batches.append(c)
                    j += c

                # backward
                self.backward(batch_targets)

                # update weights
                self.update_weights(update_rate, reg_coeff)

            # keep track of the cost values
            if use_cross_val:
                self.forward(cross_val_X)
                j = self.cost(self.outputs, cross_val_Y, reg_coeff)
                self.costs.append(j)
            else:
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

        # split into batches
        #batches = self._split_into_batches(len(X), 1000)

        #outputs = np.zeros((len(X), self.layer_sizes[-1]))

        #for start_index, end_index in batches:
        #    self.forward(X[start_index: end_index])
        #    outputs[start_index: end_index] = self.outputs

        #return outputs
