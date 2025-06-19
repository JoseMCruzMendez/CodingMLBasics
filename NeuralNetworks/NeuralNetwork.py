import numpy as np
import NpAutograd as npa
from typing import Literal, Callable, Union, List

class NeuralNetwork(npa.GradModule):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden: list[int] = None,
                 activation: Union[Callable[[npa.Tensor], npa.Tensor], List[Callable[[npa.Tensor], npa.Tensor]]] = npa.relu
    ):
        """Implements a basic Neural Network with one hidden layer by default.
        :param input_size: number of input features
        :param output_size: number of output features
        :param hidden: list of hidden layer sizes, defaults to [256,]
        :param activation: activation function for hidden layers, defaults to ReLU. Can also be a list of activation functions, in which case the length must be the length of hidden layers + 1.
        """
        super().__init__()
        #Initialize hidden sizes
        if hidden is None:
            hidden = [256,]
        #Handle the list vs single function case
        if isinstance(activation, Callable):
            activation = [activation for _ in range(len(hidden))]
        elif len(activation) != len(hidden):
            raise ValueError("Activation list must be of length hidden layers, or a single function for all layers.")

        #Store all of the params for optimizer use
        self.params = []

        #Make all of the linear layers
        self.layers = []
        prev = input_size
        for i, next_size in enumerate(hidden):
            next_layer = Linear(prev, next_size)
            self.params.append(next_layer)
            self.layers.append(next_layer)
            self.layers.append(activation[i])
            prev = next_size
        self.layers.append(Linear(prev, output_size))

    def forward(self, value: npa.Tensor, *args, **kwargs):
        if isinstance(value, np.ndarray):
            value = npa.Tensor(value)
        for layer in self.layers:
            value = layer(value)
        return value

class Linear(npa.GradModule):
    def __init__(self, in_features, out_features, init: Literal["normal", "uniform"]="normal"):
        super().__init__()
        if init == "normal":
            sampler = np.random.randn
        elif init == "uniform":
            sampler = np.random.uniform
        else:
            raise ValueError("Invalid init mode, can be normal or uniform")

        self.weights = npa.Tensor(sampler(in_features, out_features))
        self.bias = npa.Tensor(sampler(1, out_features))
        self.params = [self.weights, self.bias]

    def forward(self, value: npa.Tensor, *args, **kwargs):
        #Had issues with batch dimension I will have to correct
        return value.matmul(self.weights) + self.bias

    # def update(self, updater: Callable[[np.ndarray, np.ndarray], np.ndarray]):
    #     """Interface necessary for optimizers to update parameters"""
    #     for param in self.params:
    #         param.update_v(updater)
    #         param.zero_grad()

