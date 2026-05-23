import unittest
import numpy as np
import sys

# Set path to the module under test
sys.path.append('../SimpleNeuralNetwork')
from SimpleNeuralNetwork import SimpleNeuralNetwork

class TestSimpleNeuralNetwork(unittest.TestCase):

    def setUp(self):
        """Common setup for all tests."""
        self.input_size = 2
        self.hidden_size = 4
        self.output_size = 1
        self.network = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)

    def test_initialization(self):
        """Tests whether weights and biases are correctly initialized (shapes and ranges)."""
        self.assertEqual(self.network.weights_input_hidden.shape, (self.input_size, self.hidden_size))
        self.assertEqual(self.network.bias_hidden.shape, (1, self.hidden_size))
        self.assertEqual(self.network.weights_hidden_output.shape, (self.hidden_size, self.output_size))
        self.assertEqual(self.network.bias_output.shape, (1, self.output_size))

        # Check if weights are within the range of -0.5 to 0.5
        self.assertTrue(np.all(self.network.weights_input_hidden >= -0.5) and np.all(self.network.weights_input_hidden <= 0.5))
        self.assertTrue(np.all(self.network.weights_hidden_output >= -0.5) and np.all(self.network.weights_hidden_output <= 0.5))
        
        # Check if biases are initialized to zero
        self.assertTrue(np.all(self.network.bias_hidden == 0))
        self.assertTrue(np.all(self.network.bias_output == 0))

    def test_sigmoid(self):
        """Tests the accuracy of the sigmoid function and its behavior for extreme values."""
        test_input = np.array([-1.0, 0.0, 1.0])
        expected_output = np.array([0.26894142, 0.5, 0.73105858])
        np.testing.assert_allclose(self.network.sigmoid(test_input), expected_output, atol=1e-8)

        # Added from the second set: testing extreme values (saturation)
        extreme_input = np.array([[-100.0, 100.0]])
        extreme_output = self.network.sigmoid(extreme_input)
        self.assertTrue(np.all(extreme_output >= 0) and np.all(extreme_output <= 1))

    def test_sigmoid_derivative(self):
        """Tests the correctness of the sigmoid function derivative."""
        test_output = np.array([0.2, 0.5, 0.8])
        expected_derivative = np.array([0.16, 0.25, 0.16])
        np.testing.assert_allclose(self.network.sigmoid_derivative(test_output), expected_derivative, atol=1e-8)

    def test_forward_propagation(self):
        """Tests forward propagation (output shape and storage of internal states)."""
        input_data = np.array([[1.0, 0.5]])  # Two-dimensional array for consistency with matrices
        output = self.network.forward_propagation(input_data)

        # Check output shape
        self.assertEqual(output.shape, (1, self.output_size))

        # The output of the sigmoid function must be between 0 and 1
        self.assertTrue(np.all(output >= 0) and np.all(output <= 1))

        # Check if internal attributes are correctly saved for backpropagation
        self.assertTrue(hasattr(self.network, 'hidden_layer_input'))
        self.assertTrue(hasattr(self.network, 'hidden_layer_output'))
        self.assertTrue(hasattr(self.network, 'output_layer_input'))
        self.assertTrue(hasattr(self.network, 'output'))
        
        self.assertEqual(self.network.hidden_layer_input.shape, (1, self.hidden_size))
        self.assertEqual(self.network.hidden_layer_output.shape, (1, self.hidden_size))
        self.assertEqual(self.network.output_layer_input.shape, (1, self.output_size))
        self.assertEqual(self.network.output.shape, (1, self.output_size))

    def test_backward_propagation_shapes(self):
        """Tests whether backpropagation changes the shape of weights and biases."""
        input_data = np.array([[1.0, 0.5]])
        target = np.array([[0.9]])

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

    def test_train_method_smoke_test(self):
        """Smoke test to check if the train method runs without throwing exceptions during a few iterations."""
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = np.array([[0], [1], [1], [0]])
        try:
            self.network.train(inputs, targets, num_iterations=10)
        except Exception as e:
            self.fail(f"Training method threw an exception: {e}")

    def test_predict_method(self):
        """Tests whether the predict method returns an output of the correct shape and range."""
        input_data = np.array([[1.0, 0.0]])
        prediction = self.network.predict(input_data)
        self.assertEqual(prediction.shape, (1, self.output_size))
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1))

    def test_xor_training(self):
        """Added from the second set: Tests whether the network can actually learn the XOR function."""
        # Set the seed before initialization or training (if your network doesn't use a seed internally).
        # Ideally, set the seed directly in SimpleNeuralNetwork if supported,
        # or overwrite the weights with a specific seed:
        np.random.seed(42)  # Try numbers like 42, 1, or 100
        
        # If weights are generated in __init__, re-create the network for this test with a fixed seed:
        self.network = SimpleNeuralNetwork(self.input_size, self.hidden_size, self.output_size)

        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = np.array([[0], [1], [1], [0]])
        
        # If your class allows it, try increasing the learning rate, e.g., to 0.5 or 1.0:
        # self.network.learning_rate = 0.5 
        
        # Increasing the number of iterations to 10,000 ensures better convergence for poorer initializations
        self.network.train(inputs, targets, num_iterations=10000)
        
        predictions = np.array([self.network.predict(inp.reshape(1, -1)) for inp in inputs]).reshape(4, 1)
        error = np.mean((predictions - targets) ** 2)
        
        self.assertLess(error, 0.05, f"The network failed to learn XOR. MSE is too high: {error}")

if __name__ == '__main__':
    unittest.main()