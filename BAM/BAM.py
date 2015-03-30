__author__ = 'misha'

import numpy as np

class BAM(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        self.M = np.zeros((self.input_size, self.output_size)).astype(np.int8)

    def train(self, inputs, outputs):

        if not isinstance(inputs, (list, tuple, np.ndarray)):
            print "inputs should be a list, tuple or np.ndarray"
            raise

        if not isinstance(outputs, (list, tuple, np.ndarray)):
            print "outputs should be a list, tuple or np.ndarray"
            raise
        try:
            inputs = np.array(inputs).reshape((-1, self.input_size))
            outputs = np.array(outputs).reshape((-1, self.output_size))
        except:
            print "Something is wrong with the inputs or outputs"

        if len(inputs) != len(outputs):
            print "number of inputs is not equal to the number of outputs"
            raise

        inputs = 2 * inputs - 1
        outputs = 2 * outputs - 1

        self.M += np.dot(inputs.T, outputs)

    def one_pass(self, tests, is_input=True, tell=False):

        size = self.input_size
        m = self.M

        if not is_input:
            size = self.output_size
            m = self.M.T

        tests = np.array(tests).reshape((-1, size))
        tests = 2 * tests - 1

        res = np.dot(tests, m)

        if tell:
            print "\n", tests
            print m
            print res, '\n'

        res[res > 0] = 1
        res[res < 0] = 0

        return res

    def retrieve(self, tests, is_input=True):
        pass
