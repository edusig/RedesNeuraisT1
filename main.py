try:
   import cPickle as pickle
except:
   import pickle
import gzip
import numpy as np
from network import Network


def load_data():
    with gzip.open('mnist.pkl.gz', 'rb') as training:
        file = pickle._Unpickler(training)
        file.encoding = 'latin1'
        training_data, validation_data, test_data = file.load()
    return (training_data, validation_data, test_data)


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    return (training_inputs, training_results, validation_inputs, va_d[1], test_inputs, te_d[1])


def vectorized_result(j):
    e = np.zeros((1, 10))
    e[0][j] = 1.0
    return e

training_inputs, training_results, validation_inputs, validation_results, test_inputs, test_results = load_data_wrapper()
n = Network([784, 300, 10])
n.SGD(training_inputs[:3000], training_results[:3000], 0.3, 5000)
for i in range(5):
    k = np.random.randint(len(validation_inputs))
    result = n.process(validation_inputs[i])
    print("CASE #{}:\n input[{}] = {} \n\n Expected = {}\n\n\n".format(i, k, result, validation_results[k]))