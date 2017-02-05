# Load pickled data
import pickle
import os


def load_data():
    # TODO: Fill this in based on where you saved the training and testing data
    directory = os.path.dirname(__file__)
    training_file1 = os.path.join(directory, 'traffic-signs-data/train.p')
    training_file = os.path.abspath('./train.p')

    testing_file = os.path.abspath('./test.p')

    with open(training_file, 'r') as f:
        train = pickle.loads(f)
    with open(testing_file, 'r') as f:
        test = pickle.loads(f)

    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']


if __name__ == "__main__":
    load_data()
