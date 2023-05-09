from modules import SimpleSelfAttention, MHSelfAttention
from torch import nn
import utils


class MultiheadModel(nn.Module):
    """Multihead attention model with positional encoding."""

    def __init__(self, vocab_size, n_classes=2, k=256, pool='avg', heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)  # token embedding

        self.encoding = utils.pos_encode(k)  # positional encoding

        self.attention = MHSelfAttention(k, heads)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # encoded: (batch_size, seq_len, embedding_dim)
        encoded = embedded + self.encoding[:embedded.size(1), :].unsqueeze(0)

        # attended: (batch_size, seq_len, embedding_dim)
        attended = self.attention(encoded)
        # swap the position of the embedding and time dimension so that we can apply the pooling layer
        attended = attended.permute(0, 2, 1)

        # pooled: (batch_size, embedding_dim, 1)
        pooled = self.pooling(attended)
        # pooled: (batch_size, embedding_dim)
        pooled = pooled.view(pooled.size(0), -1)

        # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        return self.linear(pooled)


class SimpleModel(nn.Module):
    """Simple attention model without positional encoding and a single self-attention head."""

    def __init__(self, vocab_size, n_classes=2, k=256, pool='avg'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)

        self.attention = SimpleSelfAttention(k)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        # attended: (batch_size, seq_len, embedding_dim)
        attended = self.attention(embedded)
        # swap the position of the embedding and time dimension so that we can apply the pooling layer
        attended = attended.permute(0, 2, 1)

        # pooled: (batch_size, embedding_dim, 1)
        pooled = self.pooling(attended)
        # pooled: (batch_size, embedding_dim)
        pooled = pooled.view(pooled.size(0), -1)

        # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        projected = self.linear(pooled)

        return projected


class BaseModel(nn.Module):
    """The base classification model consisting of an embedding layer, one global pooling operation (max or avg)
    and a linear projection from the embedding dimension down to the number of classes."""

    def __init__(self, vocab_size, n_classes=2, k=256, pool='avg'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        # swap the position of the embedding and time dimension so that we can apply the pooling layer
        embedded = embedded.permute(0, 2, 1)

        # pooled: (batch_size, embedding_dim, 1)
        pooled = self.pooling(embedded)
        # remove the extra dimension: (batch_size, embedding_dim)
        pooled = pooled.squeeze(2)

        # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        projected = self.linear(pooled)

        return projected
