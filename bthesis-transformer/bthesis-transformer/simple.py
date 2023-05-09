import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Simple self-attention layer with and weight normalisation."""
    def __init__(self, k):
        super().__init__()
        
        self.k = k

        self.toQueries = nn.Linear(k, k, bias=False)  # bias=False so that we can use this as a simple projection
        self.toKeys = nn.Linear(k, k, bias=False)
        self.toValues = nn.Linear(k, k, bias=False)
    
    def forward(self, x):
        b, t, k = x.size()

        queries = self.toQueries(x)
        keys = self.toKeys(x)
        values = self.toValues(x)

        # compute raw attention scores (dot product attention)
        dot_raw = torch.bmm(queries, keys.transpose(1,2))  # (b, t, t)

        # normalise the raw attention scores
        dot_scaled = dot_raw / (queries.size(1) ** (1/2))  # (b, t, t) | using queries.size(1) instead of k so that scaling works for any embedding dimension

        # row-wise softmax
        dot = F.softmax(dot_scaled, dim=2)  # (b, t, t)

        # apply the attention weights to the values
        return torch.bmm(dot, values).view(b, t, k)
 

class Transformer(nn.Module):
    def __init__(self, vocab_size, n_classes=2, k=512, pool='avg'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)

        self.attention = SelfAttention(k)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")
        
        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)

        attended = self.attention(embedded)  # attended: (batch_size, seq_len, embedding_dim)
        attended = attended.permute(0, 2, 1)  # swap the position of the embedding and time dimension so that we can apply the pooling layer

        pooled = self.pooling(attended)  # pooled: (batch_size, embedding_dim, 1)
        pooled = pooled.view(pooled.size(0), -1)  # pooled: (batch_size, embedding_dim)
        
        projected = self.linear(pooled)  # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        
        return projected