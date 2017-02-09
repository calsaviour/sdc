# Load pickled data
import numpy as np
import pandas as pd
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def MSE(y, Y):
    return np.mean((y - Y) ** 2)


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes  # 3
        self.hidden_nodes = hidden_nodes  # 2
        self.output_nodes = output_nodes  # 1

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                        (self.hidden_nodes, self.input_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.output_nodes ** -0.5,
                                                         (self.output_nodes, self.hidden_nodes))
        self.lr = learning_rate

        #### Set this to your implemented sigmoid function ####
        # Activation function is the sigmoid function
        self.activation_function = sigmoid

    def train(self, inputs_list, targets_list):
        # Convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer
        hidden_inputs = self.weights_input_to_hidden.dot(inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = self.weights_hidden_to_output.dot(hidden_outputs)  # signals into final output layer
        final_outputs = final_inputs  # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        output_errors = targets - self.activation_function(
                final_outputs)  # Output layer error is the difference between desired target and actual output.

        # TODO: Backpropagated error
        deltaOutputLayer = sigmoid_derivative(final_outputs) * output_errors  # errors propagated to the hidden layer
        hiddenOutputChanges = self.lr * hidden_outputs.dot(deltaOutputLayer)

        deltaHiddenLayer = deltaOutputLayer.dot(self.weights_hidden_to_output).dot(sigmoid_derivative(hidden_inputs))
        inputHiddenChanges = self.lr * inputs.dot(deltaHiddenLayer)

        # TODO: Update the weights
        self.weights_hidden_to_output -= hiddenOutputChanges.T  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden -= inputHiddenChanges.T  # update input-to-hidden weights with gradient descent step

    # hidden_errors = sigmoid_derivative(final_outputs) * output_errors  # errors propagated to the hidden layer
    # hidden_grad = sigmoid_derivative(hidden_inputs) * (self.weights_hidden_to_output.T.dot(hidden_errors))
    #
    # # hidden_grad = hidden_errors.dot(self.weights_hidden_to_output) * hidden_outputs.T.dot((1 - hidden_outputs))
    #
    # # TODO: Update the weights
    # self.weights_hidden_to_output = self.weights_hidden_to_output + (self.lr * (hidden_outputs.dot(
    #         targets).T))  # update hidden-to-output weights with gradient descent step
    # self.weights_input_to_hidden = self.weights_input_to_hidden + (self.lr * hidden_grad.dot(
    #         inputs.T))  # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = self.weights_input_to_hidden.dot(inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = hidden_outputs  # signals into final output layer
        final_outputs = self.weights_hidden_to_output.dot(final_inputs)  # signals from final output layer

        return final_outputs


if __name__ == "__main__":
    data_path = 'Bike-Sharing-Dataset/hour.csv'
    rides = pd.read_csv(data_path)
    rides.head()
    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    data.head()
    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean) / std
    # Save the last 21 days
    test_data = data[-21 * 24:]
    data = data[:-21 * 24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    # Hold out the last 60 days of the remaining data as a validation set
    train_features, train_targets = features[:-60 * 24], targets[:-60 * 24]
    val_features, val_targets = features[-60 * 24:], targets[-60 * 24:]
    np.random.normal(0.0, 1 ** -0.5, (1, 2))

    ### Set the hyperparameters here ###
    epochs = 5
    learning_rate = 0.1
    hidden_nodes = 4
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train': [], 'validation': []}
    for e in range(epochs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        for record, target in zip(train_features.ix[batch].values,
                                  train_targets.ix[batch]['cnt']):
            network.train(record, target)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
        sys.stdout.write("\rProgress: " + str(100 * e / float(epochs))[:4] \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Validation loss: " + str(val_loss)[:5])

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)
