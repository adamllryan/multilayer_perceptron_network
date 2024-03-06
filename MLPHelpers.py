import numpy as np
from typing import Callable


class ActivationFunction:
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        func = ActivationFunction.sigmoid
        return func(x) * (1 - func(x))

    # def tanh(x):
    # def tanh_derivative(x):
    # def relu(x):
    # def relu_derivative(x):
    # def leaky_relu(x):
    # def leaky_relu_derivative(x):
    # def softmax(x):
    # def softmax_derivative(x):

    def get(name: str) -> callable:
        if name == "sigmoid":
            return (ActivationFunction.sigmoid, ActivationFunction.sigmoid_derivative)
        else:
            raise ValueError("Activation function not found")


class CostFunction:
    def mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
        return 0.5 * np.mean(np.square(actual - predicted))

    def mean_squared_error_derivative(
        actual: np.ndarray, predicted: np.ndarray
    ) -> np.ndarray:
        return predicted - actual

    # def cross_entropy(actual, predicted):
    # def cross_entropy_derivative(actual, predicted):
    # def mean_absolute_error(actual, predicted):
    # def mean_absolute_error_derivative(actual, predicted):

    def get(name: str) -> Callable:
        if name == "mean_squared_error":
            return (
                CostFunction.mean_squared_error,
                CostFunction.mean_squared_error_derivative,
            )
        else:
            raise ValueError("Cost function not found")


class Regularization:
    # def l1(weights: np.ndarray, lambda_: float) -> float:
    # def l2(weights: np.ndarray, lambda_: float) -> float:
    def none(weights: np.ndarray, lambda_: float) -> float:
        return 0

    def get(name: str) -> callable:
        if name == "none":
            return Regularization.none
        else:
            raise ValueError("Regularization function not found")


class Optimizer:

    def sgd(error: np.ndarray, activator: Callable, weighted_sum: np.ndarray):
        return -error * activator(weighted_sum)

    def get(name: str) -> callable:
        if name == "sgd":
            return Optimizer.sgd
        else:
            raise ValueError("Optimizer function not found")


class LearningRateScheduler:
    def constant(learning_rate: float, *args):
        return learning_rate

    def stochastic_approximation(learning_rate: float, iteration: int, *args):
        return learning_rate / iteration

    def search_then_converge(x: float, iteration: int, slope: float, *args):
        return x / (1 + iteration / slope)

    def get(name: str) -> callable:
        if name == "constant":
            return LearningRateScheduler.constant
        elif name == "stochastic_approximation":
            return LearningRateScheduler.stochastic_approximation
        elif name == "search_then_converge":
            return LearningRateScheduler.search_then_converge
        else:
            raise ValueError("Learning rate scheduler not found")


class Normalization:

    # def batch
    # def layer

    def get(name: str) -> callable:
        if name == "none":
            return None
        else:
            raise ValueError("Normalization function not found")


class Initialization:

    def random(
        weights: list[int], seed: int = None, min: float = -1, max: float = 1
    ) -> list[np.ndarray]:
        np.random.seed(seed)
        return [
            np.random.uniform(min, max, (weights[i + 1], weights[i] + 1))
            for i in range(len(weights) - 1)
        ]

    # def xavier
    # def he

    def get(name: str) -> callable:
        if name == "random":
            return Initialization.random
        else:
            raise ValueError("Initialization function not found")


class WeightUpdate:
    def none(
        weights: np.ndarray, gradients: np.ndarray, learning_rate: float
    ) -> np.ndarray:
        return weights - learning_rate * gradients

    def momentum(
        weights: np.ndarray,
        gradients: np.ndarray,
        learning_rate: float,
        momentum: float,
        previous_update: np.ndarray,
    ) -> np.ndarray:
        return weights - learning_rate * gradients + momentum * previous_update

    def get(name: str) -> callable:
        if name == "none":
            return WeightUpdate.none
        elif name == "momentum":
            return WeightUpdate.momentum
        else:
            raise ValueError("Weight update function not found")
