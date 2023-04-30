import torch
from torch import nn
from torch.nn import functional as F

class SelfAttention(nn.Module):
    """Multi-head self-attention layer with and weight normalisation."""
    def __init__(self, emb, heads=4):  # emb is the dimensionality of the embedding space (len of the input vector)
        super().__init__()
        
        assert emb % heads == 0  # embedding dimension must be divisible by number of heads
        self.k, self.heads = emb, heads

        # computing queries, keys and values in parallel for all heads
        self.toQueries = nn.Linear(emb, emb, bias=False)  # bias=False so that we can use this as a simple projection
        self.toKeys = nn.Linear(emb, emb, bias=False)
        self.toValues = nn.Linear(emb, emb, bias=False)

        self.unifyHeads = nn.Linear(emb, emb)  # W0 matrix
    
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
    def __init__(self, vocab_size, n_classes=2, emb_dim=512, pooling='avg', heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.attention = SelfAttention(emb_dim, heads)

        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")
        
        self.linear = nn.Linear(emb_dim, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)

        attended = self.attention(embedded)  # attended: (batch_size, seq_len, embedding_dim)
        attended = attended.permute(0, 2, 1)  # swap the position of the embedding and time dimension so that we can apply the pooling layer

        pooled = self.pooling(attended)  # pooled: (batch_size, embedding_dim, 1)
        pooled = pooled.view(pooled.size(0), -1)  # pooled: (batch_size, embedding_dim)
        
        projected = self.linear(pooled)  # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        
        return projected