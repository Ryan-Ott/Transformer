import torch
from torch import nn
from torch.nn import functional as F


class MHSelfAttention(nn.Module):
    """Multi-head self-attention"""

    # k is the dimensionality of the embedding space (len of the input vector)
    def __init__(self, k, heads=4, mask=False):
        super().__init__()

        assert k % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads, self.mask = k, heads, mask

        # computing queries, keys and values in parallel for all heads
        # bias=False so that we can use this as a simple projection
        self.toQueries = nn.Linear(k, k, bias=False)
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

        self.unifyHeads = nn.Linear(k, k)  # W0 matrix

    def forward(self, x):
        b, t, e = x.size()
        h = self.heads

        assert e == self.k  # sanity check

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        s = e // h  # s is the dimensionality of the embedding space per head

        # split the embedding space into multiple heads
        queries = queries.view(b, t, h, s)
        keys = keys.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # fold heads into batch dimension so that we can bmm all heads at once
        # first swapping the time and head dimensions, then folding the heads into the batch dimension
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # scaling the queries and keys in place to save memory
        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))  # instead of scaling the dot product

        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b * h, t, t)

        if self.mask:  # masking out the upper half of the dot matrix, excluding the diagonal
            mask = torch.triu(torch.ones(t, t)).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            dot = dot + mask.unsqueeze(0)

        # row-wise softmax
        dot = F.softmax(dot, dim=2)  # (b * h, t, t)

        # apply the attention weights to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions back again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class SimpleSelfAttention(nn.Module):
    """Simple self-attention layer with and weight normalisation."""

    def __init__(self, k):
        super().__init__()

        self.k = k

        # bias=False so that we can use this as a simple projection
        self.toQueries = nn.Linear(k, k, bias=False)
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

    def forward(self, x):
        b, t, e = x.size()

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        # compute raw attention scores (dot product attention)
        dot_raw = torch.bmm(queries, keys.transpose(1, 2))  # (b, t, t)

        # normalise the raw attention scores
        # (b, t, t) | using queries.size(1) instead of k so that scaling works for any embedding dimension
        dot_scaled = dot_raw / (queries.size(1) ** (1/2))

        # row-wise softmax
        dot = F.softmax(dot_scaled, dim=2)  # (b, t, t)

        # apply the attention weights to the values
        return torch.bmm(dot, values).view(b, t, e)


class EncoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.attention = MHSelfAttention(k, heads, mask=False)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            # 4 can be reduced to 2 or 1, if we want to save memory
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        
        fedforward = self.ff(x)

        return self.norm2(fedforward + x)


class DecoderBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()

        self.maskedAttention = MHSelfAttention(k, heads, mask=True)
        self.attention = MHSelfAttention(k, heads, mask=False)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.norm3 = nn.LayerNorm(k)

        self.ff = nn.Sequential(
            # 4 can be reduced to 2 or 1, if we want to save memory
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k))
    
    def forward(self, x, encoder):
        masked_attended = self.maskedAttention(x)
        x = self.norm1(masked_attended + x)

        attended = self.attention(encoder(x))
        x = self.norm2(attended + x)

        fedforward = self.ff(x)
        return self.norm3(fedforward + x)
        