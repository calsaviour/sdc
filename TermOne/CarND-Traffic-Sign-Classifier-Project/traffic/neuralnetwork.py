# Load pickled data
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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
        final_outputs = self.activation_function(final_inputs)  # signals from final output layer

        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error
        output_errors = targets - final_outputs  # Output layer error is the difference between desired target and actual output.

        # TODO: Backpropagated error
        hidden_errors = final_outputs * (1 - final_outputs) * output_errors  # errors propagated to the hidden layer
        hidden_grad = hidden_errors.dot(self.weights_hidden_to_output) * hidden_outputs.T * (1 - hidden_outputs.T)

        # TODO: Update the weights
        self.weights_hidden_to_output += self.lr * (hidden_outputs.dot(
                hidden_errors).T)  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * hidden_grad.T.dot(
                inputs.T)  # update input-to-hidden weights with gradient descent step

    def run(self, inputs_list):
        # Run a forward pass through the network
        inputs = np.array(inputs_list, ndmin=2).T

        #### Implement the forward pass here ####
        # TODO: Hidden layer
        hidden_inputs = self.weights_input_to_hidden.dot(inputs)  # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

        # TODO: Output layer
        final_inputs = self.weights_hidden_to_output.dot(hidden_outputs)  # signals into final output layer
        final_outputs = self.activation_function(final_inputs)  # signals from final output layer

        return final_outputs
