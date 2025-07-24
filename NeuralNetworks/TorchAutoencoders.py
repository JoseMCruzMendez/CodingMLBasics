import torch
import torch.nn as nn
from typing import List, Union, Optional, Callable, Tuple


class TorchNN(nn.Module):
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
                layers.append(BatchNormer(size))
            layers.append(activation())
            prev_size = size
        layers.append(nn.Linear(prev_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class BatchNormer(nn.Module):
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super(BatchNormer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1,num_features))
        self.beta = nn.Parameter(torch.zeros(1, num_features))
        self.register_buffer("running_mean", torch.zeros(1, num_features))
        self.register_buffer("running_var", torch.zeros(1, num_features))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor):
        if self.training:
            cur_var, cur_mean = torch.var_mean(x, dim=0, keepdim=True, unbiased=False)
            with torch.no_grad(): #this was causing graph issues down the line
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * cur_mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * cur_var)
                mean, var = cur_mean, cur_var
        else:
            mean, var = self.running_mean, self.running_var
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.gamma * x_hat + self.beta

class TorchAutoencoder(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, encoder_shape:Optional[List[int]], decoder_shape:Optional[List[int]]) -> None:
        super(TorchAutoencoder, self).__init__()
        self.encoder = TorchNN(in_dim=in_dim, output_size=bottleneck_dim, hidden_size=encoder_shape, activation=nn.ReLU)
        self.batch_norm = BatchNormer(num_features=bottleneck_dim)
        self.decoder = TorchNN(in_dim=bottleneck_dim, output_size=in_dim, hidden_size=decoder_shape, activation=nn.ReLU)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.batch_norm(x)
        x = self.decoder(x)
        return x


class TorchVAE(nn.Module):
    def __init__(self, in_dim: int, bottleneck_dim: int, encoder_shape:Optional[List[int]], decoder_shape:Optional[List[int]]) -> None:
        super(TorchVAE, self).__init__()
        self.mean_net = TorchNN(in_dim=in_dim, output_size=bottleneck_dim, hidden_size=encoder_shape, activation=nn.ReLU)
        self.log_var_net = TorchNN(in_dim=in_dim, output_size=bottleneck_dim, hidden_size=encoder_shape, activation=nn.ReLU)

        self.decoder = TorchNN(in_dim=bottleneck_dim, output_size=in_dim, hidden_size=decoder_shape, activation=nn.ReLU)
        self.mu = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.log_var = nn.Linear(bottleneck_dim, bottleneck_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] :
        mu = self.mean_net(x)
        log_var = self.log_var_net(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        x_hat = self.decoder(z)
        return x_hat, mu, log_var, z

class LeNet5(nn.Module):
    """Implements the LeNet-5 Convolutional Architecture, no NN."""
    def __init__(self, activation=nn.ReLU, normalize=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        #Will try torch batch norm 2d for testing purposes
        self.bn1 = nn.BatchNorm2d(6) if normalize else nn.Identity()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16) if normalize else nn.Identity()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.activation = activation
        self.head = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.BatchNorm1d(120) if normalize else nn.Identity(),
            self.activation(),
            nn.Linear(120, 84),
            nn.BatchNorm1d(84) if normalize else nn.Identity(),
            self.activation(),
            nn.Linear(84, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #CONV 1 + NORM
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation()(x)
        #POOL 1
        x = self.pool1(x)
        #CONV 2 + NORM
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation()(x)
        #POOL 2
        x = self.pool2(x)
        #START DENSE LAYER
        x = x.view(-1, 16*5*5)
        x = self.head(x)
        return x

