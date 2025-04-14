import numpy as np
from SimpleNeuralNetwork import *

class LogicGateTrainer:
    """
    A class to train and test a simple neural network for logic gate operations.
    """
    def __init__(self, input_size, 
                    hidden_size, output_size, 
                    training_inputs, training_targets,
                    num_iterations=25000,
                    gate_name="Logic Gate"):
        """
        Initializes the LogicGateTrainer.

        Args:
            input_size (int): The number of input neurons.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output neurons.
            training_inputs (list of list): The training input data.
            training_targets (list of list): The corresponding training target data.
            num_iterations (int, optional): The number of training iterations (epochs). Defaults to 25000.
            learning_rate (float, optional): The learning rate for the neural network. Defaults to 0.1.
            gate_name (str, optional): A descriptive name for the logic gate. Defaults to "Logic Gate".
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.training_inputs = np.array(training_inputs)
        self.training_targets = np.array(training_targets)
        self.num_iterations = num_iterations
        self.gate_name = gate_name
        self.network = SimpleNeuralNetwork(input_size, hidden_size, output_size)

    def train(self):
        """
        Trains the neural network using the provided training data.
        """
        print(f"\n{self.gate_name}")
        print(f"Training the network for {self.num_iterations} iterations...")
        self.network.train(self.training_inputs, self.training_targets, self.num_iterations)
        print("Training complete.")

    def test(self, test_inputs):
        """
        Tests the trained neural network with the provided test inputs.

        Args:
            test_inputs (list of list): The input data to test the network with.
        """
        print("\nTesting the trained network:")
        for input_data in test_inputs:
            prediction = self.network.predict(np.array(input_data))
            print(f"Prediction for {input_data}: {prediction}")

if __name__ == "__main__":
    xor_trainer = LogicGateTrainer(
        input_size=2,
        hidden_size=4,
        output_size=1,
        training_inputs=[[0, 0], [0, 1], [1, 0], [1, 1]],
        training_targets=[[0], [1], [1], [0]],
        gate_name="XOR GATE"
    )
    xor_trainer.train()
    xor_trainer.test([[0, 0], [0, 1], [1, 0], [1, 1]])

    and_trainer = LogicGateTrainer(
        input_size=2,
        hidden_size=4,
        output_size=1,
        training_inputs=[[0, 0], [0, 1], [1, 0], [1, 1]],
        training_targets=[[0], [0], [0], [1]],
        gate_name="AND GATE"
    )
    and_trainer.train()
    and_trainer.test([[0, 0], [0, 1], [1, 0], [1, 1]])

    xor3_trainer = LogicGateTrainer(
        input_size=3,
        hidden_size=4,
        output_size=1,
        training_inputs=[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
        training_targets=[[0], [1], [1], [0], [1], [0], [0], [1]],
        gate_name="XOR GATE - 3 inputs"
    )
    xor3_trainer.train()
    xor3_trainer.test([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                       [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])