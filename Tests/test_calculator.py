import unittest
import numpy as np
import sys

# Setting the path to the module to be tested
sys.path.append('../SimpleNeuralNetwork')
from SimpleNeuralNetwork import SimpleNeuralNetwork

class TestSimpleNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """Set up for all tests."""
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 1
        self.network = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)

    def test_initialization(self):
        """Tests if weights and biases are initialized correctly."""
        self.assertEqual(self.network.weights_input_hidden.shape, (self.input_size, self.hidden_size))
        self.assertEqual(self.network.bias_hidden.shape, (1, self.hidden_size))
        self.assertEqual(self.network.weights_hidden_output.shape, (self.hidden_size, self.output_size))
        self.assertEqual(self.network.bias_output.shape, (1, self.output_size))

        self.assertTrue(np.all(self.network.weights_input_hidden >= -0.5) and np.all(self.network.weights_input_hidden <= 0.5))
        self.assertTrue(np.all(self.network.weights_hidden_output >= -0.5) and np.all(self.network.weights_hidden_output <= 0.5))
        self.assertTrue(np.all(self.network.bias_hidden == 0))
        self.assertTrue(np.all(self.network.bias_output == 0))

    def test_sigmoid(self):
        """Tests the correctness of the sigmoid function."""
        test_input = np.array([-1.0, 0.0, 1.0])
        expected_output = np.array([0.26894142, 0.5, 0.73105858])
        np.testing.assert_allclose(self.network.sigmoid(test_input), expected_output, atol=1e-8)

    def test_sigmoid_derivative(self):
        """Tests the correctness of the sigmoid derivative function."""
        test_output = np.array([0.2, 0.5, 0.8])
        expected_derivative = np.array([0.16, 0.25, 0.16])
        np.testing.assert_allclose(self.network.sigmoid_derivative(test_output), expected_derivative, atol=1e-8)

    def test_forward_propagation(self):
        """Tests the forward propagation pass."""
        input_data = np.array([1.0, 0.5]) # Changed to 2 elements
        output = self.network.forward_propagation(input_data)

        # Tests if the output has the correct shape
        self.assertEqual(output.shape, (1, self.output_size))

        # We cannot precisely test the numerical values of the output because the weights are random.
        # However, we can check if the output values are between 0 and 1 (due to the sigmoid function).
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))

        # Tests if internal attributes are set
        self.assertTrue(hasattr(self.network, 'hidden_layer_input'))
        self.assertTrue(hasattr(self.network, 'hidden_layer_output'))
        self.assertTrue(hasattr(self.network, 'output_layer_input'))
        self.assertTrue(hasattr(self.network, 'output'))
        self.assertEqual(self.network.hidden_layer_input.shape, (1, self.hidden_size))
        self.assertEqual(self.network.hidden_layer_output.shape, (1, self.hidden_size))
        self.assertEqual(self.network.output_layer_input.shape, (1, self.output_size))
        self.assertEqual(self.network.output.shape, (1, self.output_size))

    def test_backward_propagation_shapes(self):
        """Tests if backward propagation does not change the shape of weights and biases."""
        input_data = np.array([1.0, 0.5]) # Changed to 2 elements
        target = np.array([0.9]) # Changed to 1 element

        initial_weights_input_hidden = self.network.weights_input_hidden.copy()
        initial_bias_hidden = self.network.bias_hidden.copy()
        initial_weights_hidden_output = self.network.weights_hidden_output.copy()
        initial_bias_output = self.network.bias_output.copy()

        self.network.forward_propagation(input_data)
        self.network.backward_propagation(input_data, target)

        self.assertEqual(self.network.weights_input_hidden.shape, initial_weights_input_hidden.shape)
        self.assertEqual(self.network.bias_hidden.shape, initial_bias_hidden.shape)
        self.assertEqual(self.network.weights_hidden_output.shape, initial_weights_hidden_output.shape)
        self.assertEqual(self.network.bias_output.shape, initial_bias_output.shape)

    def test_train_method(self):
        """Tests if the training method runs without errors."""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = np.array([[0], [1], [1], [0]])
        try:
            self.network.train(inputs, targets, num_iterations=10)
            self.assertTrue(True) # If no exception occurs, the test is successful in this regard
        except Exception as e:
            self.fail(f"Training method raised an exception: {e}")

    def test_predict_method(self):
        """Tests if the prediction method returns output of the correct shape."""
        input_data = np.array([1.0, 0.0]) # Changed to 2 elements
        prediction = self.network.predict(input_data)
        self.assertEqual(prediction.shape, (1, self.output_size))
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1))

if __name__ == '__main__':
    unittest.main()