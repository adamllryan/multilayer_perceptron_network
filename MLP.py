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

    class SingleCore:
        class ActivationFunction:
            def sigmoid(x: np.ndarray) -> np.ndarray:
                return 1 / (1 + np.exp(-x))

            def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
                x = 1 / (1 + np.exp(-x))
                return x * (1 - x)

            activation_functions = {
                "sigmoid": (
                    sigmoid,
                    sigmoid_derivative,
                ),
                # "tanh": (tanh, tanh_derivative),
                # "relu": (relu, relu_derivative),
                # "leaky_relu": (leaky_relu, leaky_relu_derivative)
            }

        class CostFunction:
            def mean_squared_error(
                actual: np.ndarray, predicted: np.ndarray
            ) -> np.ndarray:
                return 0.5 * np.mean((actual - predicted) ** 2)

            def mean_squared_error_derivative(
                actual: np.ndarray, predicted: np.ndarray
            ) -> np.ndarray:
                return predicted - actual

            cost_functions = {
                "mean_squared_error": (
                    mean_squared_error,
                    mean_squared_error_derivative,
                ),
                # "mean_absolute_error": mean_absolute_error,
                # "cross_entropy": cross_entropy
            }

        class Regularization:
            regularization_methods = {
                # "l1": l1,
                # "l2": l2,
                # "dropout": dropout
            }

        class Optimizer:
            def sgd(error: np.ndarray, activator: Callable, weighted_sum: np.ndarray):
                return -error * activator(weighted_sum)

            optimizers = {
                "sgd": sgd,
                # "momentum": momentum,
                # "adam": adam,
                # "rmsprop": rmsprop
            }

        class LearningRateScheduler:
            def constant(x, *args):
                return x

            def stochastic_approximation(x, iterration, *args):
                return x / iterration

            # def predetermined(x, *args):

            def search_then_converge(x, iterration, slope, *args):
                return x / (1 + iterration / slope)

            learning_rate_schedulers = {
                "constant": constant,
                "stochastic_approximation": stochastic_approximation,
                "search_then_converge": search_then_converge,
                # "time_based": LearningMode.time_based,
                # "step_decay": LearningMode.step_decay,
                # "exponential_decay": LearningMode.exponential_decay
            }

        class Normalization:
            normalization_methods = {
                # "batch": batch_norm,
                # "layer": layer_norm
            }

        class Initialization:
            def random(weights: list[int], seed, min, max) -> np.ndarray:
                np.random.seed(seed)
                return [
                    np.random.uniform(min, max, (weights[i + 1], weights[i] + 1))
                    for i in range(len(weights) - 1)
                ]

            initialization_methods = {
                "random": random
                # "xavier": xavier_init,
                # "he": he_init,
            }

        class WeightUpdate:
            def no_momentum(
                weights: np.ndarray, gradients: np.ndarray, learning_rate: float
            ) -> np.ndarray:
                return weights - learning_rate * gradients

            # def momentum(weights: np.ndarray, gradients: np.ndarray, momentums: np.ndarray, learning_rate: float, prev_momentum: float) -> np.ndarray:
            #    return weights - learning_rate * gradients + prev_momentum

            weight_update_methods = {
                "no_momentum": no_momentum,
                # "momentum": momentum
            }

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

    def train(self, x: np.ndarray, y: np.ndarray) -> None:
        weighted_sums = []

        """Forward Pass (minus the activation function)"""

        previous_output = np.atleast_2d(np.append(x, 1)).T

        layers = self.weights  # readablility

        for layer in layers:

            weighted_sums.append(np.dot(layer, previous_output))

            """ we need to forward pass but we only care about the sums """

            previous_output = np.append(self.activation_function(weighted_sums[-1]), 1)

        """ Backward Pass """

        """ start by computing the error at the output layer """
        """ E = loss_function(y, y_hat) """
        """ LMS = \sum_{i=1}^{n} (y_i - y_hat_i)^2 """

        error = self.loss_function(y, previous_output[:-1])

        """ Then use delta rule to compute the error at output layer """
        """ \delta = (y - y_hat) * \phi'(z) """


if __name__ == "__main__":
    layers = [4, 4, 1]
    mlp = MLPNetwork(layers)
    mlp.display()
    # print(mlp.predict(np.random.rand(layers[0])))
    print("Model predicts: ", mlp.predict(np.array([0, 0, 0, 0])))
