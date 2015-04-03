import cPickle
import gzip


def load_data():
    with gzip.open('', 'rb') as training:
        training_data, validation_data, test_data = cPickle.load(training)
    return (training_data, validation_data, test_data)
