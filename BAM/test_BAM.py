__author__ = 'misha'

import numpy as np
import sys
from BAM import BAM



def get_random_examples(size, n_examples):

    arr = np.empty((0,size)).astype(np.int8)

    for i in range(n_examples):

        candidate = (np.random.rand(size) < 0.2) * 1

        while np.dot(candidate, arr.T).sum() > 0 or candidate.sum() == 0 or candidate.sum() > size / n_examples:
            candidate = (np.random.rand(size) < 0.2) * 1

        arr = np.concatenate((arr, candidate.reshape((-1, size))))

    return arr


if __name__ == '__main__':


    in_size = 15
    out_size = 10
    n_examples = 3

    if len(sys.argv) > 2:
        in_size = int(sys.argv[1])
        out_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_examples = int(sys.argv[3])

    b = BAM(in_size, out_size)

    inputs = get_random_examples(in_size, n_examples)
    outputs = get_random_examples(out_size, n_examples)

    print "Inputs:"
    print inputs
    print "Outputs:"
    print outputs

    b.train(inputs, outputs)

    test = [(np.random.rand(in_size) < 0.5) * 1 for i in range(1)]
    print "Test: ", test
    print "Correlations with train inputs: ", np.dot(test, inputs.T) / inputs.sum(axis=1).astype(float)
    res = b.one_pass(test)
    print "Test output: ", res
    print "correlations with train outputs: ", np.dot(res, outputs.T) / outputs.sum(axis=1).astype(float)

    back = b.one_pass(res, is_input=False)
    print "Back: ", back
    print "correlations with train input: ", np.dot(back, inputs.T) / inputs.sum(axis=1).astype(float)

    print b.iterate(test)