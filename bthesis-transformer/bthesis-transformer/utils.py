import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class Model(nn.Module):
    """A simple model that embeds the input sequences, applies max or average pooling and projects the embedding vectors down to the number of classes."""
    def __init__(self, vocab_size, n_classes=2, embedding_dim=128, pooling='max'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # TODO: add self attention layer

        self.pooling = pooling
        self.linear = nn.Linear(embedding_dim, n_classes, bias=True)  # should bias be True or False?

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)

        if self.pooling == 'max':
            pooled = torch.max(embedded, dim=1)[0]  # pooled: (batch_size, embedding_dim)
        elif self.pooling == 'avg':
            pooled = torch.mean(embedded, dim=1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")
        
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
    loss_fn = nn.CrossEntropyLoss()  # check documentation for order of batch dimension

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


#-------------------------------helper functions-------------------------------#

def visualize_weights(weight_matrix):
    """Visualize the attention weights as a heatmap."""
    plt.figure(figsize=(10, 10))
    sns.heatmap(weight_matrix, cmap='coolwarm', center=0)
    plt.title('Attention weights')
    plt.xlabel('Input sequence')
    plt.ylabel('Output sequence')
    plt.show()


def visualize_embeddings(embedding_matrix, num_points=500, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity)
    points = tsne.fit_transform(embedding_matrix[:num_points])

    plt.scatter(points[:, 0], points[:, 1])
    plt.show()

def get_tensor_memory(tensor):
    return tensor.element_size() * tensor.nelement()


def get_memory_usage_and_token_count_for_batches(batches_x, batches_y):
    memory_usages_and_token_counts = []
    for batch_x, batch_y in zip(batches_x, batches_y):
        memory_usage = get_tensor_memory(batch_x) + get_tensor_memory(batch_y)
        token_count = batch_x.numel()  # Count all elements, including padding tokens
        memory_usages_and_token_counts.append((memory_usage, token_count))
    return memory_usages_and_token_counts
