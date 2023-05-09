import math
import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """Multi-head self-attention layer with and weight normalisation."""
    def __init__(self, k, heads=4):  # k is the dimensionality of the embedding space (len of the input vector)
        super().__init__()
        
        assert k % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads = k, heads

        # computing queries, keys and values in parallel for all heads
        self.toQueries = nn.Linear(k, k, bias=False)  # bias=False so that we can use this as a simple projection
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

        self.unifyHeads = nn.Linear(k, k)  # W0 matrix
    
    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        s = k // h  # s is the dimensionality of the embedding space per head
        
        # split the embedding space into multiple heads
        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # fold heads into batch dimension so that we can bmm all heads at once
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)  # first swapping the time and head dimensions, then folding the heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # compute raw attention scores
        dot_raw = torch.bmm(queries, keys.transpose(1,2))  # (b * h, t, t)

        # normalise the raw attention scores
        dot_scaled = dot_raw / (queries.size(1) ** (1/2))  # (b * h, t, t)

        # row-wise softmax
        dot = F.softmax(dot_scaled, dim=2)  # (b * h, t, t)

        # apply the attention weights to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions back again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_classes=2, k=512, pool='avg', heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)  # token embedding
        
        self.encoding = self.encode(k)  # positional encoding

        self.attention = SelfAttention(k, heads)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")
        
        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)
        encoded = embedded + self.encoding[:embedded.size(1), :].unsqueeze(0)  # encoded: (batch_size, seq_len, embedding_dim)

        attended = self.attention(encoded)  # attended: (batch_size, seq_len, embedding_dim)
        attended = attended.permute(0, 2, 1)  # swap the position of the embedding and time dimension so that we can apply the pooling layer

        pooled = self.pooling(attended)  # pooled: (batch_size, embedding_dim, 1)
        pooled = pooled.view(pooled.size(0), -1)  # pooled: (batch_size, embedding_dim)
        
        return self.linear(pooled)  # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes

    def encode(self, k, max_len=10000):
        """Computes positional encoding for a sequence of length max_len and dimensionality k. Based on the formula from the Attention is All You Need paper."""
        pos = torch.arange(0, max_len).unsqueeze(1)  # pos: (max_len, 1)
        dim = torch.exp(torch.arange(0, k, 2) * (-math.log(10000.0) / k))  # dim: (k/2)
        enc = torch.zeros(max_len, k)  # enc: (max_len, k)
        enc[:, 0::2] = torch.sin(pos * dim)  # filling the even columns of the embedding matrix with sin(pos * dim)
        enc[:, 1::2] = torch.cos(pos * dim)  # filling the odd columns of the embedding matrix with cos(pos * dim)

        return enc