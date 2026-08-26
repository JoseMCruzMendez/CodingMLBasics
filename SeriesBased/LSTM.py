import torch
import torch.nn as nn
from RNN import TorchNN

class LSTM(nn.Module):
    def __init__(self, in_dim, output_size, latent_size):
        super(LSTM, self).__init__()
        self.latent_size = latent_size
        #self.encoder = TorchNN(in_dim, latent_size)
        self.decoder = TorchNN(latent_size, output_size)
        self.forgetter = nn.Sequential(
            nn.Linear(in_dim + latent_size, latent_size),
            nn.Sigmoid()
        )
        self.keeper = nn.Sequential(
            nn.Linear(in_dim + latent_size, latent_size),
            nn.Sigmoid()
        )
        self.candidate_net = nn.Sequential(
            nn.Linear(in_dim + latent_size, latent_size),
            nn.Tanh()
        )
        self.output_net = nn.Sequential(
            nn.Linear(in_dim + latent_size, latent_size),
            nn.Sigmoid()
        )

    def forward(self, x, hidden_state=None, cell_state=None):
        batch_size, seq_len, _ = x.shape
        if hidden_state is None:
            hidden_state = torch.zeros((batch_size, self.latent_size))
        if cell_state is None:
            cell_state = torch.zeros((batch_size, self.latent_size))
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat((x_t, hidden_state), dim=1)
            forgetting_mask = self.forgetter(combined)
            keeping_mask = self.keeper(combined)
            unmasked_candidate = self.candidate_net(combined)
            output_mask = self.output_net(combined)
            cell_state = forgetting_mask * cell_state + keeping_mask * unmasked_candidate
            hidden_state = output_mask * torch.tanh(cell_state)
            output = self.decoder(hidden_state)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden_state, cell_state