# Load pickled data
import Pickle
import os


def load_data():
    # TODO: Fill this in based on where you saved the training and testing data

    training_file = os.path.abspath('./train.p')

    testing_file = os.path.abspath('./test.p')

    with open(training_file, mode='rb') as f:
        train = cPickle.loads(f)
    with open(testing_file, mode='rb') as f:
        test = cPickle.loads(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']


if __name__ == "__main__":
    load_data()
