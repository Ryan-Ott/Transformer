import torch
from torch import nn
from . import utils, modules


class SumTransformer(nn.Module):
    """Text summarization transformer."""
    def __init__(self, device, emb_dim, vocab_size, max_len, enc_heads, enc_hidden, enc_dropout, enc_depth, dec_heads, dec_hidden, dec_dropout, dec_depth):
        super().__init__()

        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim).to(device)
        self.pos_embedding = nn.Embedding(num_embeddings=max_len, embedding_dim=emb_dim).to(device)

        self.encoder = [modules.EncoderBlock(emb_dim, enc_heads, mask=False, hidden=enc_hidden, dropout=enc_dropout) for _ in range(enc_depth)]
        self.decoder = [modules.DecoderBlock(emb_dim, dec_heads, dec_hidden, dec_dropout) for _ in range(dec_depth)]

        # self.encoder = nn.Sequential(*[modules.EncoderBlock(emb_dim, enc_heads, mask=False, hidden=enc_hidden, dropout=enc_dropout) for _ in range(enc_depth)])
        # self.decoder = nn.Sequential(*[modules.DecoderBlock(emb_dim, dec_heads, dec_hidden, dec_dropout) for _ in range(dec_depth)])  # ! nn.Sequential() doesn't work here because more than just the previous layer's output is needed as input to a layer

        self.toProbs = nn.Linear(emb_dim, vocab_size)  # convert to probabilities over vocab


    def forward(self, source, target):
        device = source.device
        tokens_source = self.token_embedding(source.to(device))
        tokens_target = self.token_embedding(target.to(device))

        b, t_s, k = tokens_source.size()
        _, t_t, _ = tokens_target.size()

        positions_source = self.pos_embedding(torch.arange(t_s, device=device))[None, :, :].expand(b, t_s, k)
        positions_target = self.pos_embedding(torch.arange(t_t, device=device))[None, :, :].expand(b, t_t, k)

        x = tokens_source + positions_source
        for layer in self.encoder:
            x = layer(x)
        context = x

        y = tokens_target + positions_target
        for layer in self.decoder:
            y = layer(y, context)

        return self.toProbs(y)


class GrtTransformer(nn.Module):
    """Auto-regressive text generation transformer."""
    def __init__(self, k, heads, depth, seq_len, token_count):
        super().__init__()

        self.token_embedding = nn.Embedding(embedding_dim=k, num_embeddings=token_count)
        self.pos_embedding = nn.Embedding(embedding_dim=k, num_embeddings=seq_len)

        blocks = [modules.EncoderBlock(k, heads, mask=True) for _ in range(depth)]
        self.encoder = nn.Sequential(*blocks)

        self.toProbs = nn.Linear(k, token_count)  # convert to probabilities over vocab

        self.norm = nn.LayerNorm(k)

    def forward(self, x):
        """
        Forward pass for the transformer.
        :param x: input tensor of shape [batch_size, seq_len]
        :return: output tensor of shape [batch_size, seq_len, token_count]
        """
        seq_len = x.shape[1]

        # Token embedding
        tokens = self.token_embedding(x)
        
        # Position embedding
        positions = torch.arange(seq_len, device=x.device)  # (seq_len)
        positions = self.pos_embedding(positions)  # (seq_len, k)
        
        # Combine token and position embeddings
        x = tokens + positions.unsqueeze(0).expand_as(tokens)  # (batch_size, seq_len, k)
        
        # Pass through the transformer blocks
        x = self.encoder(x)  # (batch_size, seq_len, k)

        # Normalize
        x = self.norm(x)  # (batch_size, seq_len, k)
        # TODO: check if normalising here is necessary because of layer norm in EncoderBlock
        
        # Convert to probabilities
        return self.toProbs(x)  # (batch_size, seq_len, token_count)


class ClfTransformer(nn.Module):
    """Text classification transformer."""
    def __init__(self, vocab_size, n_classes=2, k=512, pool='avg', max_len=5000, heads=4, depth=4):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, k)

        self.pos_encoding = utils.pos_encode(k, max_len)
        # self.pos_embedding = nn.Embedding(max_len, k)

        blocks = [modules.EncoderBlock(k, heads, mask=False) for _ in range(depth)]
        self.encoder = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(0.1)

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        self.norm = nn.LayerNorm(k)
        
        self.toProbs = nn.Linear(k, n_classes, bias=True)
    
    def forward(self, x):  # x: (batch_size, seq_len)
        tokens = self.tok_embedding(x)

        positions = self.pos_encoding[:tokens.size(1), :].unsqueeze(0).to(tokens.device)

        x = tokens + positions

        x = self.encoder(x)

        x = self.pooling(x.transpose(1, 2)).squeeze(2)  # (batch_size, k)

        x = self.norm(x)

        return self.toProbs(x)


class MultiheadClf(nn.Module):
    """Multihead attention model with positional encoding."""

    def __init__(self, vocab_size, n_classes=2, k=256, pool='avg', heads=4, device=torch.device('cpu')):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, k).to(device)  # token embedding

        self.pos_encoding = utils.pos_encode(k).to(device)  # positional encoding

        self.attention = modules.MHSelfAttention(k, heads, mask=False).to(device)  # TODO: ask whether these .to(device) are necessary

        if pool == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        self.linear = nn.Linear(k, n_classes, bias=True)

    def forward(self, x):  # x: (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.tok_embedding(x)
        # encoded: (batch_size, seq_len, embedding_dim)
        encoded = embedded + self.pos_encoding[:embedded.size(1), :].unsqueeze(0)

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


class SimpleClf(nn.Module):
    """Simple attention model without positional encoding and a single self-attention head."""

    def __init__(self, vocab_size, n_classes=2, k=256, pool='avg'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, k)

        self.attention = modules.SimpleSelfAttention(k)

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
        return self.linear(pooled)


class BaseClf(nn.Module):
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
        return self.linear(pooled)
