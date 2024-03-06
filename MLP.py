import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Callable


# TODO:
# 1. Implement the MLP class
# 2. Variance Initialization
# 3. Activation Functions, add support for sigmoid, tanh, relu, and leaky relu


class MLPNetwork:
    class NeuralNetworkPlugins:
        class ActivationFunction:
            def sigmoid(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def tanh(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def tanh_derivative(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def relu(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def relu_derivative(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def leaky_relu(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def leaky_relu_derivative(x: np.ndarray) -> np.ndarray:
                raise NotImplementedError

        class CostFunction:
            def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def mean_absolute_error(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
                raise NotImplementedError

            def cross_entropy(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
                raise NotImplementedError

        class Regularization:
            def l1(weights: np.ndarray, regularization_rate: float) -> np.ndarray:
                raise NotImplementedError

            def l2(weights: np.ndarray, regularization_rate: float) -> np.ndarray:
                raise NotImplementedError

            def dropout(weights: np.ndarray, dropout_rate: float) -> np.ndarray:
                raise NotImplementedError

        class Optimizer:
            def sgd(
                weights: np.ndarray, gradients: np.ndarray, learning_rate: float
            ) -> np.ndarray:
                raise NotImplementedError

            def momentum(
                weights: np.ndarray,
                gradients: np.ndarray,
                momentums: np.ndarray,
                learning_rate: float,
                momentum: float,
            ) -> np.ndarray:
                raise NotImplementedError

            def adam(
                weights: np.ndarray,
                gradients: np.ndarray,
                momentums: np.ndarray,
                learning_rate: float,
                momentum: float,
            ) -> np.ndarray:
                raise NotImplementedError

            def rmsprop(
                weights: np.ndarray,
                gradients: np.ndarray,
                momentums: np.ndarray,
                learning_rate: float,
                momentum: float,
            ) -> np.ndarray:
                raise NotImplementedError

        class LearningMode:
            def batch(x, *args):
                raise NotImplementedError

            def mini_batch(x, *args):
                raise NotImplementedError

            def stochastic(x, *args):
                raise NotImplementedError

        class Normalization:
            def batch_norm(x, *args):
                raise NotImplementedError

            def layer_norm(x, *args):
                raise NotImplementedError

        class Initialization:
            def xavier_init(shape: Tuple[int, int], seed, min, max) -> np.ndarray:
                raise NotImplementedError

            def he_init(shape: Tuple[int, int], seed, min, max) -> np.ndarray:
                raise NotImplementedError

            def random(shape: Tuple[int, int], seed, min, max) -> np.ndarray:
                raise NotImplementedError

        class WeightUpdate:
            def no_momentum(
                weights: np.ndarray, gradients: np.ndarray, learning_rate: float
            ) -> np.ndarray:
                raise NotImplementedError

            def momentum(
                weights: np.ndarray,
                gradients: np.ndarray,
                momentums: np.ndarray,
                learning_rate: float,
                prev_momentum: float,
            ) -> np.ndarray:
                raise NotImplementedError

    class SingleCore(NeuralNetworkPlugins):
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
    processing_type: NeuralNetworkPlugins

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
        multi_processing: bool = False,
        output_labels: List[str] = None,
        cost_function: str = "mean_squared_error",
    ):
        """Assign multiprocessing"""

        if multi_processing:
            raise NotImplementedError("Multi Processing not yet implemented")

        self.processing_type = self.MultiCore if multi_processing else self.SingleCore

        """Asserts"""

        assert len(shape) > 2, "There must be at least 3 layers in the network. "
        assert all([isinstance(i, int) for i in shape]), "All layers must be integers. "
        assert all([i > 0 for i in shape]), "All layers must be greater than zero. "
        assert (
            learning_rate_scheduler
            in self.processing_type.LearningRateScheduler.learning_rate_schedulers.keys()
        ), f'Learning mode "{learning_rate_scheduler}" not supported'
        assert (
            init_method
            in self.processing_type.Initialization.initialization_methods.keys()
        ), f'Initialization method "{init_method}" not supported'
        assert (
            activation_function
            in self.processing_type.ActivationFunction.activation_functions.keys()
        ), f'Activation function "{self.activation_function}" not supported'
        assert (
            cost_function in self.processing_type.CostFunction.cost_functions.keys()
        ), f'Cost function "{cost_function}" not supported'

        """ Network Settings """

        self.layers = shape
        self.procesing_type = self.MultiCore if multi_processing else self.SingleCore

        if output_labels is not None:
            assert (
                len(output_labels) == shape[-1]
            ), "Output labels must match the number of output nodes. "
            self.output_labels = output_labels

        """ Learning Rate Handling """

        self.learning_rate_scheduler = (
            self.processing_type.LearningRateScheduler.learning_rate_schedulers[
                learning_rate_scheduler
            ]
        )
        self.learning_rate = learning_rate

        """ Weight Initialization handling """

        self.init_method = self.processing_type.Initialization.initialization_methods[
            init_method
        ]
        self.weights = self.init_method(self.layers, seed, min, max)

        """ Activation Function Handling """

        (
            self.activation_function,
            self.activation_fuction_derivative,
        ) = self.processing_type.ActivationFunction.activation_functions[
            activation_function
        ]

        """ Momentum handling """

        if (
            momentum > 0
        ):  # having different weight update functions saves us from having to check for
            # momentum in the update method
            self.last_momentum = [
                self.init_method((shape[i], shape[i + 1]))
                for i in range(len(shape) - 1)
            ]

            self.momentum = momentum
            self.weight_update_function = self.SingleCore.WeightUpdate.momentum
        else:
            self.weight_update_function = self.SingleCore.WeightUpdate.no_momentum

        """ Cost Function Handling """

        (self.loss_function, self.loss_function_derivative) = (
            self.processing_type.CostFunction.cost_functions[cost_function]
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
