import numpy as np
from SimpleNeuralNetwork import *

# Example usage of the neural network
if __name__ == "__main__":
    print ("XOR GATE")
    # Define the training data for a logical XOR gate
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [1], [1], [0]])

    # Create an instance of the SimpleNeuralNetwork
    network = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    num_training_iterations = 100000
    print(f"Training the network for {num_training_iterations} iterations...")
    network.train(training_inputs, training_targets, num_training_iterations)
    print("Training complete.")

    # Test the trained network with the training data
    print("\nTesting the trained network:")
    print(f"Prediction for [0, 0]: {network.predict(np.array([0, 0]))}")
    print(f"Prediction for [0, 1]: {network.predict(np.array([0, 1]))}")
    print(f"Prediction for [1, 0]: {network.predict(np.array([1, 0]))}")
    print(f"Prediction for [1, 1]: {network.predict(np.array([1, 1]))}")


    print ("\nAND GATE")
    # Define the training data for a logical XOR gate
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [0], [0], [1]])

    # Create an instance of the SimpleNeuralNetwork
    network = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    num_training_iterations = 100000
    print(f"Training the network for {num_training_iterations} iterations...")
    network.train(training_inputs, training_targets, num_training_iterations)
    print("Training complete.")

    # Test the trained network with the training data
    print("\nTesting the trained network:")
    print(f"Prediction for [0, 0]: {network.predict(np.array([0, 0]))}")
    print(f"Prediction for [0, 1]: {network.predict(np.array([0, 1]))}")
    print(f"Prediction for [1, 0]: {network.predict(np.array([1, 0]))}")
    print(f"Prediction for [1, 1]: {network.predict(np.array([1, 1]))}")
