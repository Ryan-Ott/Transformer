import data_rnn
import torch
import torch.nn as nn

'''#TODO:
1. Load all the reviews with their labels and sort them by length.
2. Write a loop that slices out batches of reviews (first by a fixed number of instances per batch, then later by a fixed number
of tokens per batch).
3. Pad all instances within a batch to be the same length.
4. Build a classification model consisting of one embedding layer and a global pooling operation.
In the first case use max pooling, in the second case use average pooling.
5. Train the model and test different global pooling methods.
'''
# Create a model with an embedding layer and a global pooling operation
class Model(nn.Module):
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


def batch_by_batchsize(sequences, labels, batch_size=32):
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
    
    sequences = sequences[:n_batches * batch_size]  # Truncate to a multiple of batch_size
    labels = labels[:n_batches * batch_size]

    for batch_idx in range(n_batches):
        batch = sequences[batch_idx * batch_size: (batch_idx + 1) * batch_size]
        max_len = max(len(x) for x in batch)

        for seq in batch:
            seq += [PAD] * (max_len - len(seq))  # Pad all instances within a batch to be the same length
        
        batches_x.append(batch)
        batches_y.append(labels[batch_idx * batch_size: (batch_idx + 1) * batch_size])
    return batches_x, batches_y


def batch_by_tokencount(sequences, labels, max_tokens=1024):  # TODO: fix this to get closer to the max_tokens value
    """Create variable-size batches of sequences by grouping them by total number of tokens.
    Note that the function may create batches that are smaller than the specified `num_tokens`
    if a sequence is longer than `num_tokens`. In this case, the sequence is split into
    multiple smaller sequences that fit into the batch, each of which is padded separately.

    Args:
        sequences (List): Input sequences
        labels (List): Corresponding output labels
        num_tokens (int, optional): Maximum number of tokens in a batch. Defaults to 1024.

    Returns:
        tuple: Batches of padded input sequences and their corresponding output labels.
    """    
    batches_sequences = []
    batches_labels = []

    current_sequences = []
    current_labels = []
    current_tokens = 0
    max_seq_len = 0

    for seq, label in zip(sequences, labels):
        seq_len = len(seq)
        if current_tokens + seq_len * (len(current_sequences) + 1) > max_tokens:
            # Pad sequences in the current batch
            padded_batch_sequences = [s + [PAD] * (max_seq_len - len(s)) for s in current_sequences]
            batches_sequences.append(padded_batch_sequences)
            batches_labels.append(current_labels)

            # Reset for the next batch
            current_sequences = []
            current_labels = []
            current_tokens = 0
            max_seq_len = 0

        # Add the sequence and label to the current batch
        current_sequences.append(seq)
        current_labels.append(label)
        current_tokens += seq_len
        max_seq_len = max(max_seq_len, seq_len)

    # Process the last batch if it's not empty
    if current_sequences:
        padded_batch_sequences = [s + [PAD] * (max_seq_len - len(s)) for s in current_sequences]
        batches_sequences.append(padded_batch_sequences)
        batches_labels.append(current_labels)

    return batches_sequences, batches_labels


# Load the IMDb dataset
(x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=False)
PAD = w2i['.pad']  # Index of padding token (its 0)

# Sort the training data by length
x_train, y_train = zip(*sorted(zip(x_train, y_train), key=lambda x: len(x[0])))

# Create batches
x_train, y_train = batch_by_batchsize(x_train, y_train)
x_val, y_val = batch_by_batchsize(x_val, y_val)

# Set hyperparameters
ITERATIONS = 10
LEARNING_RATE = 0.001

# Create an instance of the model wirh max pooling
model_max = Model(len(i2w), pooling='max')

# Train the model
optimizer = torch.optim.Adam(model_max.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print('Training with max pooling')
for epoch in range(ITERATIONS):
    for x, y in zip(x_train, y_train):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        optimizer.zero_grad()
        y_pred = model_max(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} Loss: {loss.item():.4f}')

# Create an instance of the model with average pooling
model_avg = Model(len(i2w), pooling='avg')

# Train the model
optimizer = torch.optim.Adam(model_avg.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print('Training with average pooling')
for epoch in range(ITERATIONS):
    for x, y in zip(x_train, y_train):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)

        optimizer.zero_grad()
        y_pred = model_avg(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1} Loss: {loss.item():.4f}')

# Evaluate the model
model_max.eval()
model_avg.eval()

correct_max = 0
correct_avg = 0

for x, y in zip(x_val, y_val):
    x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    y_pred_max = torch.argmax(model_max(x), dim=1)
    y_pred_avg = torch.argmax(model_avg(x), dim=1)

    correct_max += (y_pred_max == y).sum().item()
    correct_avg += (y_pred_avg == y).sum().item()

print(f'Accuracy with max pooling: {correct_max / len(x_val):.4f}')
print(f'Accuracy with average pooling: {correct_avg / len(x_val):.4f}')