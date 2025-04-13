import numpy as np

class SimpleNeuralNetwork:
    """
    A basic feedforward neural network with one hidden layer.
    It uses the sigmoid activation function.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the neural network with specified layer sizes and random weights.

        Args:
            input_size (int): The number of input neurons.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons.

        Initializes:
            self.weights_input_hidden (numpy.ndarray): Weights connecting the input layer to the hidden layer.
                                                     Initialized with random values between -0.5 and 0.5.
            self.bias_hidden (numpy.ndarray): Biases for the hidden layer neurons, initialized to zeros.
            self.weights_hidden_output (numpy.ndarray): Weights connecting the hidden layer to the output layer.
                                                      Initialized with random values between -0.5 and 0.5.
            self.bias_output (numpy.ndarray): Biases for the output layer neurons, initialized to zeros.
        """
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        """
        The sigmoid activation function.

        Args:
            x (numpy.ndarray): The input to the sigmoid function.

        Returns:
            numpy.ndarray: The output of the sigmoid function, between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        The derivative of the sigmoid function.

        Args:
            x (numpy.ndarray): The output of the sigmoid function.

        Returns:
            numpy.ndarray: The derivative of the sigmoid function at the given input.
        """
        return x * (1 - x)

    def forward_propagation(self, input_data):
        """
        Performs the forward pass through the neural network.

        Args:
            input_data (numpy.ndarray): The input data for the network.

        Returns:
            numpy.ndarray: The output of the neural network.

        Updates internal attributes:
            self.hidden_layer_input (numpy.ndarray): The weighted sum of inputs and biases for the hidden layer.
            self.hidden_layer_output (numpy.ndarray): The output of the hidden layer after applying the sigmoid function.
            self.output_layer_input (numpy.ndarray): The weighted sum of hidden layer outputs and biases for the output layer.
            self.output (numpy.ndarray): The output of the output layer after applying the sigmoid function.
        """
        # Hidden layer calculation
        self.hidden_layer_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Output layer calculation
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_layer_input)

        return self.output

    def backward_propagation(self, input_data, target):
        # Error of the output layer
        output_error = target - self.output
        output_delta = output_error * self.sigmoid_derivative(self.output)

        # Error of the hidden layer
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        learning_rate = 0.1
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += input_data.reshape(1, -1).T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, inputs, targets, num_iterations):
        """
        Trains the neural network using the provided input and target data.

        Args:
            inputs (numpy.ndarray): The training input data.
            targets (numpy.ndarray): The corresponding target output data.
            num_iterations (int): The number of training iterations (epochs).

        For each iteration and each training sample, it performs a forward pass
        and then a backward pass to update the network's weights and biases.
        """
        for _ in range(num_iterations):
            for i in range(len(inputs)):
                output = self.forward_propagation(inputs[i])
                self.backward_propagation(inputs[i], targets[i])

    def predict(self, input_data):
        """
        Performs a prediction using the trained neural network.

        Args:
            input_data (numpy.ndarray): The input data for which to make a prediction.

        Returns:
            numpy.ndarray: The predicted output from the neural network.
        """
        return self.forward_propagation(input_data)