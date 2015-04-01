__author__ = 'misha'

import numpy as np

class BAM(object):

    def __init__(self, input_size, output_size):

        self.input_size = input_size
        self.output_size = output_size

        # declare variables
        self.inputs = None
        self.outputs = None

        self.input_sums = None
        self.output_sums = None

        # initialize memory matrix
        self.M = np.zeros((self.input_size, self.output_size)).astype(np.int8)

    def train(self, inputs, outputs):
        '''
        Trains BAM by summing up all outer products of input-output pairs
        :param inputs: array (list) of n input/source examples. Length of each example should be input_size
        :param outputs: array (list) of n output/target examples. Length of each example should be output_size
        '''

        if not isinstance(inputs, (list, tuple, np.ndarray)):
            raise Exception("inputs should be a list, tuple or np.ndarray")

        if not isinstance(outputs, (list, tuple, np.ndarray)):
            raise Exception("outputs should be a list, tuple or np.ndarray")

        # convert provided inputs and outputs into 2-d np.ndarray
        inputs = np.array(inputs).reshape((-1, self.input_size))
        outputs = np.array(outputs).reshape((-1, self.output_size))

        if len(inputs) != len(outputs):
            raise Exception("number of inputs is not equal to the number of outputs")

        # training
        self.M += np.dot(inputs.T, outputs)

        # save training examples
        self.inputs = inputs
        self.outputs = outputs

        # calculate their sums for future use
        self.input_sums = self.inputs.sum(axis=1).astype(float)
        self.output_sums = self.outputs.sum(axis=1).astype(float)

    def one_pass(self, tests, is_input=True, threshold=0.9, tell=False):
        '''
        For a given test inputs (outputs) activations calculates outputs (inputs) activations
        :param tests: array (list) of m tests. Length of each test must be input_size if is_input=True and
                output_size if is_input=False
        :param is_input: Input/output indicator, default is True
        :param threshold: threshold for calculating target activations:
                1 for anything above max * threshold and 0 otherwise
        :param tell: default False, set this to True if want intermediate information to be printed in the stdout
        :return: target activations
        '''

        size = self.input_size
        m = self.M

        if not is_input:
            size = self.output_size
            m = self.M.T

        tests = np.array(tests).reshape((-1, size))

        res = np.dot(tests, m)

        if tell:
            print "\n", res
            print res.max(axis=1)
            print res.sum(axis=1)

        threshold_val = threshold * res.max(axis=1)
        res = (res >= threshold_val[:, None]) * 1

        if tell:
            print "\n", threshold_val
            print res.sum(axis=1)
            print '\n'


        return res

    def iterate(self, tests, threshold=0.9, adapt_rate=0.02, max_iter=10):
        '''
        Iterates forward and backward propagation of tests through trained BAM.

        :param tests: array (list) of m tests. Length of each test must be input_size.
        :param threshold: Initial threshold for calculating target activations:
                1 for anything above max * threshold and 0 otherwise
        :param adapt_rate: Rate at which threshold can adapt
        :param max_iter: Maximum number of forward-backward passes
        :return: dict with results log
        '''

        # thresholds for input and output can be different
        th_in = threshold * np.ones(len(tests))
        th_out = threshold * np.ones(len(tests))

        # variable for calculated activations on the input side after one forward-backward pass
        in_back = np.array(tests).reshape((-1, self.input_size))
        # variable for calculated activations on the output side after forward pass
        out = np.zeros((len(in_back), self.output_size))

        # create empty results log
        results = dict(in_max_corr = [],
                       in_best_pattern = [],
                       in_sum = [],
                       in_threshold = [],
                       out_max_corr = [],
                       out_best_pattern = [],
                       out_sum = [],
                       out_threshold = []
        )

        # counter and arrays to track changes between iterations
        count = 0
        diff_in = np.array([1])
        diff_out = np.array([1])

        # main loop
        # stop when converged on something or if reached max number of iteration
        while (np.dot(diff_in, diff_in.T) > 0).any() and (np.dot(diff_out, diff_out.T) > 0).any() and count < max_iter:

            # starting inputs are now what came back after previous forward-backward passes
            in_in = in_back

            # calculate and record current results for the input side
            corr = (np.dot(in_in, self.inputs.T) / self.input_sums)
            results['in_max_corr'].append(corr.max(axis=1))
            results['in_best_pattern'].append(corr.argmax(axis=1))
            in_sum = in_in.sum(axis=1)
            results['in_sum'].append(in_sum)
            results['in_threshold'].append(th_in)

            # adapt input-side threshold
            th_in *= (1. - (in_sum < 30) * adapt_rate * 30. / in_sum + (in_sum > 60) * adapt_rate * in_sum / 60.)


            # reset output-side difference from previous iteration
            diff_out = out

            # calculate new output activation
            out = self.one_pass(in_in, threshold=th_out, is_input=True)

            # calculate new output-side difference from previous iteration
            diff_out -= out

            # calculate and record current results for the output side
            corr = (np.dot(out, self.outputs.T) / self.output_sums)
            results['out_max_corr'].append(corr.max(axis=1))
            results['out_best_pattern'].append(corr.argmax(axis=1))
            out_sum = out.sum(axis=1)
            results['out_sum'].append(out_sum)
            results['out_threshold'].append(th_out)

            # adapt output-side threshold
            th_out *= (1. - (out_sum < 30) * adapt_rate * 30. / out_sum + (out_sum > 60) * adapt_rate * out_sum / 60.)


            # calculate returned to the input-side activations
            in_back = self.one_pass(out, threshold=th_in, is_input=False)

            # calculate input-side difference from previous iteration
            diff_in = in_back - in_in

            # increment counter
            count += 1

        # repeat forward-backward one last time (just to show in results that it's the same as previous)
        in_in = in_back

        corr = (np.dot(in_in, self.inputs.T) / self.input_sums)
        results['in_max_corr'].append(corr.max(axis=1))
        results['in_best_pattern'].append(corr.argmax(axis=1))
        results['in_sum'].append(in_in.sum(axis=1))

        out = self.one_pass(in_in, threshold=th_out, is_input=True)

        corr = (np.dot(out, self.outputs.T) / self.output_sums)
        results['out_max_corr'].append(corr.max(axis=1))
        results['out_best_pattern'].append(corr.argmax(axis=1))
        results['out_sum'].append(out.sum(axis=1))

        results['in_threshold'].append(th_in)
        results['out_threshold'].append(th_out)


        return results


