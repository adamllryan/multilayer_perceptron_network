import numpy as np
from MLPHelpers import (
    ActivationFunction,
    CostFunction,
    Regularization,
    Optimizer,
    LearningRateScheduler,
    Normalization,
    Initialization,
    WeightUpdate,
)
import multiprocessing as mp
from typing import List, Tuple, Callable


# TODO:
# 1. Implement the MLP class
# 2. Variance Initialization
# 3. Activation Functions, add support for sigmoid, tanh, relu, and leaky relu


class MLPNetwork:

    epochs: int
    learning_rate: float
    momentum_rate: float

    # functions

    activation_function: Callable
    activation_fuction_derivative: Callable
    cost_function: Callable
    cost_function_derivative: Callable
    weight_update_function: Callable
    learning_rate_scheduler: Callable
    initialization_method: Callable

    # storage

    layers: List[int]
    weights: List[np.ndarray]
    last_momentum: List[np.ndarray]
    output_labels: List[str]

    # settings

    multi_processing: bool
    shape: List[int]

    # methods

    def __init__(
        self,
        shape: List[int],
        learning_rate: float = 1.0,
        momentum: float = 0.0,
        seed: int = 1000,
        min: float = -1,
        max: float = 1,
        learning_rate_scheduler: str = "constant",
        init_method: str = "random",
        activation_function: str = "sigmoid",
        cost_function: str = "mean_squared_error",
    ):
        # some basic asserts

        assert len(shape) > 2, "There must be at least 3 layers in the network. "
        assert all([isinstance(i, int) for i in shape]), "All layers must be integers. "
        assert all([i > 0 for i in shape]), "All layers must be greater than zero. "

        # assign layers to be our shape

        self.layers = shape

        # assign our learning rate method

        self.learning_rate_scheduler = LearningRateScheduler.get(
            learning_rate_scheduler
        )
        self.learning_rate = learning_rate

        # decide how we want to initialize our weights

        self.init_method = Initialization.get(init_method)
        self.weights = self.init_method(self.layers, seed, min, max)

        # assign our activation function / derivative

        (
            self.activation_function,
            self.activation_fuction_derivative,
        ) = ActivationFunction.get(activation_function)

        # assign momentum / weight update functions

        if momentum > 0:
            self.last_momentum = [
                self.init_method((shape[i], shape[i + 1]))
                for i in range(len(shape) - 1)
            ]

            self.weight_update_function = WeightUpdate.get("momentum")
        else:
            self.weight_update_function = WeightUpdate.get("none")
        self.momentum = momentum

        # assign our cost function

        self.loss_function, self.loss_function_derivative = CostFunction.get(
            cost_function
        )

    def display(self) -> None:
        print(f"\nLayers: {self.layers}\n\nWeights:")

        # Weights

        print(f"\nInput Layer: {np.zeros((self.layers[0]))}\n")
        for i in range(len(self.weights) - 1):
            print(f"Hidden layer {i+1}: \n{self.weights[i]}\n")
        print(f"Output layer: \n{self.weights[-1]}\n\n")

        # Rates

        print(f"Learning Rate: {self.learning_rate}")
        print(f"Momentum: {self.momentum}")

    def predict(self, input: np.ndarray) -> np.ndarray:
        """Forward Pass"""

        # we need to transpose input because it is loaded as horizontal vector

        previous_output = np.atleast_2d(np.append(input, 1)).T

        for layer in range(len(self.weights)):
            """Compute the weighted sum of the input and the weights here"""
            """ \sum_{i=1}^{n} x_i * w_i """

            weighted_sum = np.dot(self.weights[layer], previous_output)

            """ Apply the activation function to the weighted sum """
            """ \phi(\sum_{i=1}^{n} x_i * w_i) """

            previous_output = np.append(self.activation_function(weighted_sum), 1)

        return previous_output[:-1]  # because bias is always appended

    def _train_iteration(self, x: np.ndarray, y: np.ndarray) -> None:
        weighted_sums = []

        # compute the forward pass and store the weighted sums

        previous_output = np.atleast_2d(np.append(x, 1)).T

        layers = self.weights  # readablility

        for layer in layers:

            weighted_sums.append(np.dot(layer, previous_output))

            # we only care about the weighted sum

            previous_output = np.append(self.activation_function(weighted_sums[-1]), 1)

        # compute the backward pass, or derivative of the cost function

        error = [self.loss_function_derivative(y, previous_output[:-1])]
        for layer in layers[:-1][::-1]:
            error.append(self.loss_function_derivative(error[-1], layer))
        error = error[::-1]  # reverse because we did this backwards

        # weight update

        for i in range(len(layers)):
            self.weights[i] = self.weight_update_function(
                layers[i],
                error[i],
                self.learning_rate,
            )

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000) -> None:
        for _ in range(epochs):
            self._train_iteration(x, y)


if __name__ == "__main__":
    layers = [4, 4, 4, 1]
    mlp = MLPNetwork(layers)
    mlp.display()
    print("Model predicts: ", mlp.predict(np.array([0, 0, 0, 0])))
    mlp.train(np.array([0, 0, 0, 0]), np.array([0]), 100000)
    print("Model predicts: ", mlp.predict(np.array([0, 0, 0, 0])))
