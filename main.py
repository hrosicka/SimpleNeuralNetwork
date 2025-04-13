import numpy as np
from SimpleNeuralNetwork import *

# Example usage of the neural network
if __name__ == "__main__":
    print ("XOR GATE")
    # Define the training data for a logical XOR gate
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [1], [1], [0]])

    # Create an instance of the SimpleNeuralNetwork
    network_2xor = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    num_training_iterations = 25000
    print(f"Training the network for {num_training_iterations} iterations...")
    network_2xor.train(training_inputs, training_targets, num_training_iterations)
    print("Training complete.")

    # Test the trained network with the training data
    print("\nTesting the trained network:")
    print(f"Prediction for [0, 0]: {network_2xor.predict(np.array([0, 0]))}")
    print(f"Prediction for [0, 1]: {network_2xor.predict(np.array([0, 1]))}")
    print(f"Prediction for [1, 0]: {network_2xor.predict(np.array([1, 0]))}")
    print(f"Prediction for [1, 1]: {network_2xor.predict(np.array([1, 1]))}")


    print ("\nAND GATE")
    # Define the training data for a logical AND gate
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[0], [0], [0], [1]])

    # Create an instance of the SimpleNeuralNetwork
    network_2and = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

    # Train the neural network
    num_training_iterations = 25000
    print(f"Training the network for {num_training_iterations} iterations...")
    network_2and.train(training_inputs, training_targets, num_training_iterations)
    print("Training complete.")

    # Test the trained network with the training data
    print("\nTesting the trained network:")
    print(f"Prediction for [0, 0]: {network_2and.predict(np.array([0, 0]))}")
    print(f"Prediction for [0, 1]: {network_2and.predict(np.array([0, 1]))}")
    print(f"Prediction for [1, 0]: {network_2and.predict(np.array([1, 0]))}")
    print(f"Prediction for [1, 1]: {network_2and.predict(np.array([1, 1]))}")


    print ("\nXOR GATE - 3 inputs")
    # Define the training data for a logical XOR gate - 3 inputs
    training_inputs = np.array([[0, 0, 0],
                                 [0, 0, 1],
                                 [0, 1, 0],
                                 [0, 1, 1],
                                 [1, 0, 0],
                                 [1, 0, 1],
                                 [1, 1, 0],
                                 [1, 1, 1]])
    training_targets = np.array([[0],
                              [1],
                              [1],
                              [0],
                              [1],
                              [0],
                              [0],
                              [1]])

    # Create an instance of the SimpleNeuralNetwork
    network_3xor = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=1)

    # Train the neural network
    num_training_iterations = 25000
    print(f"Training the network for {num_training_iterations} iterations...")
    network_3xor.train(training_inputs, training_targets, num_training_iterations)
    print("Training complete.")

    # Test the trained network with the training data
    print("\nTesting the trained network:")
    print(f"Předpověď pro [0, 0, 0]: {network_3xor.predict(np.array([0, 0, 0]))}")
    print(f"Předpověď pro [0, 0, 1]: {network_3xor.predict(np.array([0, 0, 1]))}")
    print(f"Předpověď pro [0, 1, 0]: {network_3xor.predict(np.array([0, 1, 0]))}")
    print(f"Předpověď pro [0, 1, 1]: {network_3xor.predict(np.array([0, 1, 1]))}")
    print(f"Předpověď pro [1, 0, 0]: {network_3xor.predict(np.array([1, 0, 0]))}")
    print(f"Předpověď pro [1, 0, 1]: {network_3xor.predict(np.array([1, 0, 1]))}")
    print(f"Předpověď pro [1, 1, 0]: {network_3xor.predict(np.array([1, 1, 0]))}")
    print(f"Předpověď pro [1, 1, 1]: {network_3xor.predict(np.array([1, 1, 1]))}")