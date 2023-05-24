import torch
from torch import nn
from torch.nn import functional as F


class MHSelfAttention(nn.Module):
    """Multi-head self-attention"""

    # k is the dimensionality of the embedding space (len of the input vector)
    def __init__(self, k, heads, mask):
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

        assert e == self.k,  f'Input embedding dimension ({e}) should match layer embedding dimension ({self.k})'

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

        if self.mask:
            # Create a mask to remove the upper half of the dot matrix, excluding the diagonal
            mask = torch.triu(torch.ones(t, t), diagonal=1).to(dot.device)
            # Set the masked positions to float('-inf') to minimize their impact on the softmax operation
            mask = mask.masked_fill(mask == 1, float('-inf'))
            # Add the mask to the dot product matrix
            dot = dot + mask.unsqueeze(0)

        # row-wise softmax
        dot = F.softmax(dot, dim=2)  # (b * h, t, t)

        # apply the attention weights to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions back again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class MHCrossAttention(nn.Module):
    """Multi-head cross-attention"""

    def __init__(self, k, heads):
        super().__init__()

        assert k % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads = k, heads

        self.toQueries = nn.Linear(k, k, bias=False)
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)

        self.unifyHeads = nn.Linear(k, k)

    def forward(self, x, context):
        b, t, e = x.size()
        _, t_context, _ = context.size()
        h = self.heads

        assert e == self.k, f'Input embedding dim ({e}) should match layer embedding dim ({self.k})'

        queries = self.toQueries(x)
        keys = self.toKeys(context)  # keys and values come from the encoder's latent space representation
        values = self.toValues(context)

        s = e // h  # dimensionality of the embedding space per head

        queries = queries.view(b, t, h, s).transpose(1, 2).contiguous().view(b * h, t, s)
        keys = keys.view(b, t_context, h, s).transpose(1, 2).contiguous().view(b * h, t_context, s)
        values = values.view(b, t_context, h, s).transpose(1, 2).contiguous().view(b * h, t_context, s)

        queries = queries / (e ** (1/4))
        keys = keys / (e ** (1/4))

        dot = torch.bmm(queries, keys.transpose(1, 2))  # (b * h, t, t_context)

        dot = F.softmax(dot, dim=2)

        out = torch.bmm(dot, values).view(b, h, t, s)

        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

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
    def __init__(self, emb, heads, mask, hidden=4, dropout=0.1):
        super().__init__()

        self.attention = MHSelfAttention(emb, heads, mask)

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(  # TODO: possibly replace with GLU
            nn.Linear(emb, hidden * emb),
            nn.ReLU(),
            nn.Linear(hidden * emb, emb))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attended = self.attention(x)
        attended = self.dropout(attended)
        x = self.norm1(x + attended)
        
        fedforward = self.ff(x)
        fedforward = self.dropout(fedforward)
        x = self.norm2(x + fedforward)
        
        return x


class DecoderBlock(nn.Module):
    def __init__(self, k, heads, hidden=4, dropout=0.1):
        super().__init__()

        self.maskedAttention = MHSelfAttention(k, heads, mask=True)
        self.crossAttention = MHCrossAttention(k, heads)

        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.norm3 = nn.LayerNorm(k)

        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(  # TODO: possibly replace with GLU
            nn.Linear(k, hidden * k),
            nn.ReLU(),
            nn.Linear(hidden * k, k))
    
    def forward(self, x, context):
        masked_attended = self.maskedAttention(x)
        masked_attended = self.dropout(masked_attended)
        x = self.norm1(x + masked_attended)

        cross_attended = self.crossAttention(x, context)
        cross_attended = self.dropout(cross_attended)
        x = self.norm2(x + cross_attended)

        fedforward = self.ff(x)
        fedforward = self.dropout(fedforward)
        x = self.norm3(x + fedforward)
        
        return x