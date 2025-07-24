import numpy as np
import NpAutograd as npa
from NeuralNetworks.NeuralNetwork import NeuralNetwork, Linear
from typing import Union, List, Callable

class Autoencoder(npa.GradModule):
    def __init__(self, input_size: int, bottleneck_size: int, encoder_sizes: List[int], decoder_sizes:List[int], activation_function: Callable[[npa.Tensor], npa.Tensor] = npa.leaky_relu, var=0.001):
        super().__init__()
        self.input_size = input_size
        self.encoder = NeuralNetwork(input_size, bottleneck_size, encoder_sizes, activation_function, var=var)
        self.decoder = NeuralNetwork(bottleneck_size, input_size, decoder_sizes, activation_function, var=var)
        self.params = self.encoder.get_params() + self.decoder.get_params()

    def forward(self, x: npa.Tensor, *args, **kwargs) -> npa.Tensor:
        return self.decoder(self.encoder(x))