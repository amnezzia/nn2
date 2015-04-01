__author__ = 'misha'

import numpy as np
import pandas as pd

import sys
import time
from betterBAM import BAM



def get_random_examples(size, n_examples=100, min_num_ones=30, max_num_ones=60):

    arr = np.empty((0,size)).astype(np.int8)
    fraction = (min_num_ones + max_num_ones) * 0.5 / float(size)

    for i in range(n_examples):
        candidate = (np.random.rand(size) < fraction) * 1
        arr = np.concatenate((arr, candidate.reshape((-1, size))))

    return arr


if __name__ == '__main__':


    in_size = 11000
    out_size = 10000
    n_examples = 500

    if len(sys.argv) > 2:
        in_size = int(sys.argv[1])
        out_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_examples = int(sys.argv[3])

    b = BAM(in_size, out_size)

    t0 = time.time()
    inputs = get_random_examples(in_size, n_examples)
    t1 = time.time()
    print "Input generation time: {}".format(t1 - t0)
    sys.stdout.flush()
    outputs = get_random_examples(out_size, n_examples)
    t2 = time.time()
    print "Output generation time: {}".format(t2 - t1)
    sys.stdout.flush()

    t0 = time.time()
    b.train(inputs, outputs)
    t1 = time.time()
    print "Training time: {}".format(t1 - t0)
    sys.stdout.flush()

    input_sums = inputs.sum(axis=1).astype(float)
    output_sums = outputs.sum(axis=1).astype(float)

    for i in range(5):
        test = get_random_examples(in_size, 1)

        print "\nTest sum: {}".format(test.sum())
        print "Max correlation with train inputs: ", (np.dot(test, inputs.T) / input_sums).max()

        res = b.iterate(test, threshold=0.6, adapt_rate=0.02, max_iter=20)

        print pd.DataFrame(res)