import torch
import torch.nn as nn

from SeriesBased.RNN import TorchNN

class FFN(nn.Module):
    def __init__(self, in_dim, output_size, latent_size=256):
        super(FFN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, latent_size, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.latent_size = latent_size
        self.n_heads = n_heads
        self.head_size: int = latent_size // n_heads
        self.key_mat = nn.Linear(latent_size, latent_size)
        self.query_mat = nn.Linear(latent_size, latent_size)
        self.value_mat = nn.Linear(latent_size, latent_size)
        self.output_mat = nn.Linear(latent_size, latent_size)


    def forward(self, query, key, value):
        batch_size, seq_len, _ = key.shape
        keys = self.key_mat(key).view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1,2)
        queries = self.query_mat(query).view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1,2)
        values = self.value_mat(value).view(batch_size, seq_len, self.n_heads, self.head_size).transpose(1,2)

        attention = self.calculate_attention(keys, queries)

        x = attention @ values
        x = x.transpose(1,2).contiguous().view(batch_size, seq_len, self.latent_size)
        x = self.output_mat(x)
        return x

    def calculate_attention(self, keys, queries):
        kq_values = queries @ keys.transpose(-2, -1) / (self.head_size ** 0.5)
        attention = torch.softmax(kq_values, dim=-1)
        return attention

class MaskedMultiHeadAttention(MultiHeadAttention):
    def __init__(self, latent_size, n_heads):
        super(MaskedMultiHeadAttention, self).__init__(latent_size, n_heads)

    def calculate_attention(self, keys, queries):
        batch_size, n_heads, seq_len, _ = keys.shape
        kq_values = queries @ keys.transpose(-2, -1) / (self.head_size ** 0.5)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(keys.device)
        masked_kq_values = kq_values.masked_fill(mask, -torch.inf)
        attention = torch.softmax(masked_kq_values, dim=-1)
        return attention

class EncoderBlock(nn.Module):
    def __init__(self, latent_size, n_heads):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(latent_size, n_heads)
        self.ln1 = nn.LayerNorm(latent_size)
        self.ffn = FFN(latent_size, latent_size)
        self.ln2 = nn.LayerNorm(latent_size)

    def forward(self, x):
        x = self.attention(x, x, x) + x
        x = self.ln1(x)
        x = self.ffn(x) + x
        x = self.ln2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, latent_size, n_heads):
        super(DecoderBlock, self).__init__()
        self.self_attention = MaskedMultiHeadAttention(latent_size, n_heads)
        self.ln1 = nn.LayerNorm(latent_size)
        self.cross_attention = MultiHeadAttention(latent_size, n_heads)
        self.ln2 = nn.LayerNorm(latent_size)
        self.ffn = FFN(latent_size, latent_size)
        self.ln3 = nn.LayerNorm(latent_size)

    def forward(self, x, encoder_output):
        x = self.self_attention(x, x, x) + x
        x = self.ln1(x)
        x = self.cross_attention(x, encoder_output, encoder_output) + x
        x = self.ln2(x)
        x = self.ffn(x) + x
        x = self.ln3(x)
        return x

class EncoderTower(nn.Module):
    def __init__(self, latent_size, n_heads, n_layers):
        super(EncoderTower, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(latent_size, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DecoderTower(nn.Module):
    def __init__(self, latent_size, n_heads, n_layers):
        super(DecoderTower, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(latent_size, n_heads) for _ in range(n_layers)])

    def forward(self, x, encoder_output):
        for layer in self.layers:
            x = layer(x, encoder_output)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, in_dim, latent_size, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.positional_bias = nn.Parameter(torch.randn(max_len, latent_size) * 0.01)
        self.embedder = FFN(in_dim, latent_size)
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        positional_bias = self.positional_bias[:seq_len].unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.embedder(x) + positional_bias
        return x

class Transformer(nn.Module):
    def __init__(self, in_dim, output_size, latent_size=256, n_heads=8, n_layers=3):
        super(Transformer, self).__init__()
        self.source_position_net = PositionalEncoding(in_dim, latent_size)
        self.target_position_net = PositionalEncoding(in_dim, latent_size)
        self.encoder = EncoderTower(latent_size, n_heads, n_layers)
        self.decoder = DecoderTower(latent_size, n_heads, n_layers)
        self.output_net = FFN(latent_size, output_size)

    def forward(self, x):
        source = self.source_position_net(x)
        target = self.target_position_net(x)
        encoder_output = self.encoder(source)
        decoder_output = self.decoder(target, encoder_output)
        output = self.output_net(decoder_output)
        return output