import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):  # k is the dimensionality of the embedding space (len of the input vector)
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
        dot_raw = torch.bmm(queries, keys.transpose(1, 2))  # (b * h, t, t)

        # scale the raw attention scores
        dot = dot_raw / (k ** (1/2))

        # apply softmax to get attention weights
        dot = F.softmax(dot, dim=2)  # row-wise normalisation

        # apply attention weights to values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = SelfAttention(k, heads=heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))
        
    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)