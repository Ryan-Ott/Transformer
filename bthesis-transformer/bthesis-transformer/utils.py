import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class SelfAttention(nn.Module):
    """Multi-head self-attention layer with and weight normalisation."""
    def __init__(self, emb, heads=4, mask=False):  # emb is the dimensionality of the embedding space (len of the input vector)
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

        # scale the raw attention scores
        dot = dot_raw / (k ** (1/2))  # (b * h, t, t)

        # row-wise softmax to get normalised weights
        dot = F.softmax(dot, dim=2)  # (b * h, t, t)

        # apply the attention weights to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap head and time dimensions back again so that we can concatenate the heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * s)

        # concatenate the heads and return
        return self.unifyHeads(out)


class Transformer(nn.Module):
    def __init__(self, vocab_size, n_classes=2, emb_dim=128, pooling='avg', heads=4, mask=False):
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


def batch_by_instances(sequences, labels, batch_size=32, pad_token=0):
    """Create batches of a given number of instances and pad all instances within a batch to be the same length.

    Args:
        sequences (List): A list of input sequences
        labels (List): List of corresponding labels
        batch_size (int, optional): Number of instances in a batch. Defaults to 32.

    Returns:
        tuple: The padded input sequences and their corresponding output labels.
    """
    batches_x, batches_y = [], []

    for i in range(0, len(sequences), batch_size):
        batch_x = sequences[i:i + batch_size]
        batch_y = labels[i:i + batch_size]

        # Find the max length in the current batch
        max_len = max(len(x) for x in batch_x)

        # Pad sequences in the current batch and convert them to tensors, then stack them into a single tensor per batch
        padded_tensor_batch_x = torch.stack([torch.LongTensor(seq + [pad_token] * (max_len - len(seq))) for seq in batch_x])

        # Convert labels to tensors and stack these into a single tensor per batch
        tensor_batch_y = torch.LongTensor(batch_y)

        batches_x.append(padded_tensor_batch_x)
        batches_y.append(tensor_batch_y)

    return batches_x, batches_y


def batch_by_tokens(sequences, labels, max_tokens=4096, pad_token=0):
    def pad_and_convert_to_tensor(batch_x, batch_y, max_seq_len):
        padded_batch_x = [seq + [pad_token] * (max_seq_len - len(seq)) for seq in batch_x]
        tensor_batch_x = torch.LongTensor(padded_batch_x)
        tensor_batch_y = torch.LongTensor(batch_y)
        return tensor_batch_x, tensor_batch_y
    
    batches_x, batches_y = [], []
    batch_x, batch_y = [], []
    max_seq_len = 0

    for seq, label in zip(sequences, labels):
        seq = seq[:max_tokens] if len(seq) > max_tokens else seq

        if (len(batch_x) + 1) * max(max_seq_len, len(seq)) > max_tokens:
            tensor_batch_x, tensor_batch_y = pad_and_convert_to_tensor(batch_x, batch_y, max_seq_len)
            batches_x.append(tensor_batch_x)
            batches_y.append(tensor_batch_y)
            batch_x, batch_y = [seq], [label]
            max_seq_len = len(seq)
        else:
            batch_x.append(seq)
            batch_y.append(label)
            max_seq_len = max(max_seq_len, len(seq))

    tensor_batch_x, tensor_batch_y = pad_and_convert_to_tensor(batch_x, batch_y, max_seq_len)
    batches_x.append(tensor_batch_x)
    batches_y.append(tensor_batch_y)

    return batches_x, batches_y


def train(model, batches_x, batches_y, epochs=10, alpha=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training with {model.pooling} pooling")
    start_time = time.time()

    for epoch in range(epochs):
        for batch_x, batch_y in zip(batches_x, batches_y):
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1} Loss: {loss.item():.2f}')
    
    mins, secs = divmod(time.time() - start_time, 60)
    print(f'Training took {int(mins)}:{int(secs):02d} minutes')


def evaluate(model, batches_x, batches_y):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in zip(batches_x, batches_y):
            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f'Accuracy of the {model.pooling} pooling model: {correct / total * 100:.2f}%')


#-------------------------------verbose functions-------------------------------#

def visualize_weights(weight_matrix, title):
    """Visualize the attention weights as a heatmap."""
    plt.figure(figsize=(10, 10))
    sns.heatmap(weight_matrix, cmap='coolwarm', center=0)
    plt.title(title)
    plt.xlabel('Input sequence')
    plt.ylabel('Output sequence')
    plt.show()


def visualize_embeddings(embedding_matrix, title, num_points=500, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity)
    points = tsne.fit_transform(embedding_matrix[:num_points])

    plt.scatter(points[:, 0], points[:, 1])
    plt.title(title)
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    plt.show()


def get_memory_and_tokens_per_batch(batches_x):
    """Return a dictionary with the memory usage and token count per batch."""
    memory_tokens_per_batch = {}
    for i, batch_x in enumerate(batches_x):
        memory_usage = batch_x.element_size() * batch_x.nelement()
        token_count = batch_x.numel()  # Count all elements, including padding tokens
        memory_tokens_per_batch.setdefault(i, (memory_usage, token_count))
    return memory_tokens_per_batch


def get_memory_usage_deviation(memory_tokens_per_batch):
    """Calculate the deviation between the smallest and largest memory usage as a percentage"""
    memory_usage = [memory for memory, _ in memory_tokens_per_batch.values()]
    return (max(memory_usage) - min(memory_usage)) / min(memory_usage) * 100