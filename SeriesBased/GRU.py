import torch
import torch.nn as nn
from RNN import TorchNN

class GRU(nn.Module):
    def __init__(self, in_dim, output_size, latent_size):
        super(GRU, self).__init__()
        self.latent_size = latent_size
        self.decoder = TorchNN(latent_size, output_size)
        #Reset layer weights
        self.reset_input = nn.Linear(in_dim, latent_size, bias=False)
        self.reset_hidden = nn.Linear(latent_size, latent_size, bias=False)
        self.reset_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.reset_layer = lambda state, hidden: self.reset_bias + self.reset_hidden(hidden) + self.reset_input(state)
        #Update layer weights
        self.update_input = nn.Linear(in_dim, latent_size, bias=False)
        self.update_hidden = nn.Linear(latent_size, latent_size, bias=False)
        self.update_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.update_layer = lambda state, hidden: self.update_bias + self.update_hidden(hidden) + self.update_input(state)
        #Hidden Layer update weights
        self.hupdate_input = nn.Linear(in_dim, latent_size, bias=False)
        self.hupdate_hidden = nn.Linear(latent_size, latent_size, bias=False)
        self.hupdate_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.hupdate_layer = lambda state, hidden: self.hupdate_bias + self.hupdate_hidden(hidden) + self.hupdate_input(state)

    def forward(self, input, hidden_state=None):
        batch, seq_len, _ = input.shape
        if hidden_state is None:
            hidden_state = torch.zeros((batch, self.latent_size))
        outputs = []
        for t in range(input.shape[1]):
            x_t = input[:, t, :]
            reset_mask = torch.sigmoid(self.reset_layer(x_t, hidden_state))
            update = torch.sigmoid(self.update_layer(x_t, hidden_state))
            hidden_unmasked = torch.tanh(self.hupdate_layer(x_t, hidden_state * reset_mask))
            hidden_state = (1-update) * hidden_state + update * hidden_unmasked
            output = self.decoder(hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden_state





class expGRU(nn.Module):
    #Fixes the issue with expRNN where the exp was calculated every single computation instead of once per batch.
    def __init__(self, in_dim, output_size, latent_size):
        self.latent_size = latent_size
        super(expGRU, self).__init__()
        self.decoder = TorchNN(latent_size, output_size)
        #Reset layer weights
        self.reset_input = nn.Linear(in_dim, latent_size, bias=False)
        self.reset_hidden = nn.Parameter(torch.randn(latent_size * (latent_size + 1)//2) * 0.01)
        self.reset_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.reset_layer = lambda state, hidden, ortho: self.reset_bias + hidden@ortho + self.reset_input(state)
        #Update layer weights
        self.update_input = nn.Linear(in_dim, latent_size, bias=False)
        self.update_hidden = nn.Parameter(torch.randn(latent_size * (latent_size + 1)//2) * 0.01)
        self.update_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.update_layer = lambda state, hidden, ortho: self.update_bias + hidden@ortho + self.update_input(state)
        #Hidden Layer update weights
        self.hupdate_input = nn.Linear(in_dim, latent_size, bias=False)
        self.hupdate_hidden = nn.Parameter(torch.randn(latent_size * (latent_size + 1)//2) * 0.01)
        self.hupdate_bias = nn.Parameter(torch.randn(latent_size) * 0.01)
        self.hupdate_layer = lambda state, hidden, ortho: self.hupdate_bias + hidden@ortho + self.hupdate_input(state)

        #Ortho Helpers
        self.latent_size = latent_size
        i, j = torch.triu_indices(latent_size, latent_size)
        self.register_buffer('i', i)
        self.register_buffer('j', j)

    def orthogonalize(self, x):
        A = torch.zeros(self.latent_size, self.latent_size)
        A[self.i, self.j] = x
        A = A - A.T
        ortho_A = torch.matrix_exp(A)
        return ortho_A

    def forward(self, input, hidden_state=None):
        batch, seq_len, _ = input.shape
        if hidden_state is None:
            hidden_state = torch.zeros((batch, self.latent_size))
        outputs = []
        ortho_reset = self.orthogonalize(self.reset_hidden)
        ortho_update = self.orthogonalize(self.update_hidden)
        ortho_hupdate = self.orthogonalize(self.hupdate_hidden)
        for t in range(input.shape[1]):
            x_t = input[:, t, :]
            reset_mask = torch.sigmoid(self.reset_layer(x_t, hidden_state, ortho_reset))
            update = torch.sigmoid(self.update_layer(x_t, hidden_state, ortho_update))
            hidden_unmasked = torch.tanh(self.hupdate_layer(x_t, hidden_state * reset_mask, ortho_hupdate))
            hidden_state = (1-update) * hidden_state + update * hidden_unmasked
            output = self.decoder(hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden_state
