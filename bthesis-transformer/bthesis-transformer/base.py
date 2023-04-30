from torch import nn

class Transformer(nn.Module):
    """The base classification model consisting of an embedding layer, one global pooling operation (max or avg)
    and a linear projection from the embedding dimension down to the number of classes."""

    def __init__(self, vocab_size, n_classes=2, emb_dim=512, pooling='avg'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")
        
        self.linear = nn.Linear(emb_dim, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # swap the position of the embedding and time dimension so that we can apply the pooling layer

        pooled = self.pooling(embedded)  # pooled: (batch_size, embedding_dim, 1)
        pooled = pooled.squeeze(2)  # remove the extra dimension: (batch_size, embedding_dim)

        projected = self.linear(pooled)  # projected: (batch_size, n_classes) | project the embedding vectors down to the number of classes
        
        return projected