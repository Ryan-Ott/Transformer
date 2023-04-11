import time
import torch
import torch.nn as nn

class Model(nn.Module):
    """A simple model that embeds the input sequences and applies max or average pooling."""
    def __init__(self, vocab_size, embedding_dim=128, pooling='max'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pooling = pooling

    def forward(self, x):  # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # embedded: (batch_size, seq_len, embedding_dim)

        if self.pooling == 'max':
            pooled = torch.max(embedded, dim=1)[0]  # pooled: (batch_size, embedding_dim)
        elif self.pooling == 'avg':
            pooled = torch.mean(embedded, dim=1)
        else:
            raise ValueError("Pooling must be set to 'max' or 'avg")

        return pooled


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

    n_batches = len(sequences) // batch_size

    # Truncate to a multiple of batch_size
    sequences = sequences[:n_batches * batch_size]
    labels = labels[:n_batches * batch_size]

    for batch_idx in range(n_batches):
        batch = sequences[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        max_len = max(len(x) for x in batch)

        for seq in batch:
            seq += [pad_token] * (max_len - len(seq))

        batches_x.append(batch)
        batches_y.append(labels[batch_idx * batch_size: (batch_idx + 1) * batch_size])
    return batches_x, batches_y


def batch_by_tokens(sequences, labels, max_tokens=4096, pad_token=0):
    """Split the input sequences into batches of a given number of tokens and pad all sequences within a batch to be the same length.

    Args:
        sequences (List): Input sequences
        labels (List): Corresponding labels
        max_tokens (int, optional): Number of tokens that a batch should not exceed. Defaults to 4096.

    Returns:
        tuple: Batches of input sequences and their corresponding labels.
    """    
    batches_x, batches_y = [], []
    batch_x, batch_y = [], []
    max_seq_len = 0

    for seq, label in zip(sequences, labels):
        # if adding one more sequence would cause the batch to exceed max_tokens (including padding) close off the batch
        if (len(batch_x) + 1) * max(max_seq_len, len(seq)) > max_tokens:
            # Pad the batch
            for seq in batch_x:
                seq += [pad_token] * (max_seq_len - len(seq))

            batches_x.append(batch_x)
            batches_y.append(batch_y)
            batch_x, batch_y = [], []
            max_seq_len = 0
        
        batch_x.append(seq)
        batch_y.append(label)
        max_seq_len = max(max_seq_len, len(seq))
    
    # Pad the last batch (unclosed) batch
    for seq in batch_x:
        seq += [pad_token] * (max_seq_len - len(seq))
    
    batches_x.append(batch_x)
    batches_y.append(batch_y)
    return batches_x, batches_y


def train(model, x_train, y_train, epochs=10, alpha=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    loss_fn = nn.CrossEntropyLoss()

    print(f'\nTraining with {model.pooling} pooling')
    start_time = time.time()

    for epoch in range(epochs):
        for x, y in zip(x_train, y_train):
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1} Loss: {loss.item():.2f}')
    
    mins, secs = divmod(time.time() - start_time, 60)
    print(f'Training took {int(mins)}:{int(secs):02d} minutes')


def evaluate(model, x_val, y_val):
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in zip(x_val, y_val):
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor(y, dtype=torch.long)

            y_pred = model(x)
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f'Accuracy of the {model.pooling} pooling model: {correct / total * 100:.2f}%')

