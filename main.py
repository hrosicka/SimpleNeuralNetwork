import numpy as np
from LogicalGateTrainer import *

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