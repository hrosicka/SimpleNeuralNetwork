# Simple Neural Network

This repository contains a simple feedforward neural network implemented in Python using the NumPy library. The network has one hidden layer and utilizes the sigmoid activation function.

## Overview

The `SimpleNeuralNetwork` class provides a basic implementation of a neural network suitable for learning simple patterns. It includes methods for initialization, forward propagation, backward propagation, training, and prediction.

## Installation

No specific installation is required as the code relies only on the NumPy library. Ensure you have NumPy installed in your Python environment. If not, you can install it using pip:

```bash
pip install numpy
```
Save the SimpleNeuralNetwork.py and main.py files in the same directory.

Run the main.py script.

## Files

The repository includes the following files:

-   `main.py`: This file contains the main script that demonstrates the usage of the `SimpleNeuralNetwork` class. It trains the network on the XOR and AND logic gates with 2 inputs, and then on a 3-input XOR gate.
-   `SimpleNeuralNetwork.py`: This file contains the definition of the `SimpleNeuralNetwork` class.

## Notes
This is a very basic implementation of a neural network for educational purposes. More complex neural networks often involve multiple hidden layers, different activation functions, and more sophisticated training algorithms.

The number of training iterations and the size of the hidden layer can significantly impact the network's ability to learn. These parameters can be adjusted to improve performance.

The learning rate in the backward_propagation method is set to a fixed value of 0.1. In more advanced implementations, the learning rate might be adjusted dynamically during training.
