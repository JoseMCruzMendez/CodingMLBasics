from typing import Optional, List, Union, Callable, Literal

import torch
import torch.nn as nn

class TorchNN(nn.Module): #Like the old one but now fully torch
    def __init__(self, in_dim: int, output_size: int, hidden_size:Optional[List[int]]=None, activation:Union[Callable, List[Callable]]= nn.ReLU, normalize=True) -> None:
        super(TorchNN, self).__init__()
        self.input_size = in_dim
        self.output_size = output_size
        if hidden_size is None:
            self.hidden_size = [256,]
        else:
            self.hidden_size = hidden_size
        if isinstance(activation, Callable):
            self.activation = [activation for _ in range(len(self.hidden_size))]
        assert len(self.hidden_size) == len(self.activation), "list of activations must be the same length as hidden layers"
        prev_size = in_dim
        layers = []
        for activation, size in zip(self.activation, self.hidden_size):
            layers.append(nn.Linear(prev_size, size))
            if normalize:
                layers.append(nn.BatchNorm1d(size))
            layers.append(activation())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class RNN(nn.Module):
    def __init__(self, input_size, output_size, bottleneck_size = 128, hidden_sizes=(256, 128), output_activation=nn.Sigmoid, weight_activation = nn.Tanh ):
        super(RNN, self).__init__()
        #I ended up using a similar "Encoder latent dim" pattern as I got a little confused with the input/output recurrence
        self.bottleneck_size = bottleneck_size
        self.recurring_weight = nn.Linear(bottleneck_size, bottleneck_size)
        self.hidden_states = TorchNN(input_size, bottleneck_size, hidden_sizes)
        self.output_layer = nn.Sequential(nn.Linear(bottleneck_size, output_size), output_activation())
        self.weight_activation = weight_activation

    def forward(self, x, h_prev = None):
        #Assume x is (batch, seq_len, dim)
        batch, seq_len, _ = x.shape
        if h_prev is None:
            hidden_state = torch.zeros(batch, self.bottleneck_size)
        else: #I made a small mistake in the "forecasting" tests, as I was resetting the hidden state without realizing. Thus, the return statement
            hidden_state = h_prev
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            input_latent = self.hidden_states(x_t)
            hidden_latent = self.recurring_weight(hidden_state)
            hidden_state = self.weight_activation(input_latent + hidden_latent)
            output = self.output_layer(hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden_state

class ExpLinear(nn.Module):
    """Implements a square orthogonal transformation with a diagonal matrix. Inspired by the ExpRNN paper for training stability"""
    def __init__(self, features, init: Literal["normal", "uniform"]= "normal", var=0.001):
        super().__init__()
        if init == "normal":
            sampler = torch.randn
        elif init == "uniform":
            sampler = torch.rand
        else:
            raise ValueError("Invalid init mode, can be normal or uniform")

        var = torch.tensor(var)
        self.feature_dim = features
        self.weights = nn.Parameter(sampler(features * (features + 1) // 2) * torch.sqrt(var))
        self.bias = nn.Parameter(sampler(features) * torch.sqrt(var))
        i, j = torch.triu_indices(features, features)
        self.register_buffer('i', i)
        self.register_buffer('j', j)

    def forward(self, x):
        A = torch.zeros(self.feature_dim, self.feature_dim)
        A[self.i, self.j] = self.weights
        A = A - A.T
        ortho_A = torch.matrix_exp(A)
        return x.matmul(ortho_A) + self.bias

class ModRelu(nn.Module):
    #ModRelu activation as specified in "Cheap Orthogonal Constraints for Neural Networks" by Lezcano et Al.
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(1,1))
    def forward(self, x):
        #Assume x is (batch, dim)
        return torch.sign(x) * nn.functional.relu(x)
        #modulus = torch.norm(x, dim=1, keepdim=True)
        #modulus = torch.clamp(modulus, min=1e-6)
        #return nn.functional.relu(modulus + self.bias) * (x / modulus)

class ExpRNN(RNN):
    def __init__(self, input_size, output_size, bottleneck_size = 128, hidden_sizes=(256, 128)):
        super(ExpRNN, self).__init__(input_size, output_size, bottleneck_size, hidden_sizes)
        self.recurring_weight = ExpLinear(bottleneck_size)
        #ModRelu suggested
        self.weight_activation = ModRelu()