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

    print "Inputs autocorrelations:"
    print np.dot(inputs, inputs.T)
    print "Outputs autocorrelations:"
    print np.dot(outputs, outputs.T)

    b.train(inputs, outputs)

    test = get_random_examples(in_size, 1)
    print "Test: ", test
    print "Correlations with train: ", np.dot(test, inputs.T)
    print "Test output: ", b.one_pass(test)
    print b.one_pass([1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
    print b.one_pass([1,1,1,1,1,1,0,0,0,0], is_input=False)
