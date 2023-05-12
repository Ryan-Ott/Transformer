import math
import time
import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler


def pos_encode(k, max_len=10000):
    """Computes positional encoding for a sequence of length max_len and dimensionality k. Based on Vaswani et al."""
    pos = torch.arange(0, max_len).unsqueeze(1)  # pos: (max_len, 1)
    dim = torch.exp(torch.arange(0, k, 2) * (-math.log(10000.0) / k))  # dim: (k/2)
    enc = torch.zeros(max_len, k)  # enc: (max_len, k)

    enc[:, 0::2] = torch.sin(pos * dim)  # even columns
    enc[:, 1::2] = torch.cos(pos * dim)  # odd columns

    return enc


def sort(x, y):
    return zip(*sorted(zip(x, y), key=lambda x: len(x[0])))


def batchify(device, batch_by, x_train, y_train, x_val, y_val, PAD):
    """Create batches of the training and validation data."""
    if batch_by == 'instances':
        x_train_batches, y_train_batches = batch_by_instances(
            device, x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = batch_by_instances(
            device, x_val, y_val, pad_token=PAD)
    elif batch_by == 'tokens':
        x_train_batches, y_train_batches = batch_by_tokens(
            device, x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = batch_by_tokens(
            device, x_val, y_val, pad_token=PAD)
    else:
        raise ValueError("batch_by must be set to 'instances' or 'tokens'")

    return x_train_batches, y_train_batches, x_val_batches, y_val_batches


def batch_by_instances(device, sequences, labels, batch_size=32, pad_token=0):
    """Create batches of a given number of instances and pad all instances within a batch to be the same length.

    Args:
        device (torch.device): Device to load the tensors onto
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
        padded_tensor_batch_x = torch.stack(
            [torch.LongTensor(seq + [pad_token] * (max_len - len(seq))).to(device) for seq in batch_x])

        # Convert labels to tensors and stack these into a single tensor per batch
        tensor_batch_y = torch.LongTensor(batch_y).to(device)

        batches_x.append(padded_tensor_batch_x)
        batches_y.append(tensor_batch_y)

    return batches_x, batches_y


def batch_by_tokens(device, sequences, labels, max_tokens=2**14, pad_token=0):
    """Create batches of a maximum number of tokens so that each batch takes roughly the same amount of memory. Pad all instances within a batch to be the same length.

    Args:
        device (torch.device): Device to load the tensors onto
        sequences (List): A list of input sequences
        labels (List): List of corresponding labels
        max_tokens (int, optional): Maximum number of tokens in a batch. Defaults to 32,768.

    Returns:
        tuple: The padded input sequences and their corresponding output labels.
    """
    
    def pad_and_convert_to_tensor(batch_x, batch_y, max_seq_len):
        padded_batch_x = [seq + [pad_token] *
                          (max_seq_len - len(seq)) for seq in batch_x]
        tensor_batch_x = torch.LongTensor(padded_batch_x).to(device)
        tensor_batch_y = torch.LongTensor(batch_y).to(device)
        return tensor_batch_x, tensor_batch_y

    batches_x, batches_y = [], []
    batch_x, batch_y = [], []
    max_seq_len = 0

    for seq, label in zip(sequences, labels):
        seq = seq[:max_tokens] if len(seq) > max_tokens else seq

        if (len(batch_x) + 1) * max(max_seq_len, len(seq)) > max_tokens:
            tensor_batch_x, tensor_batch_y = pad_and_convert_to_tensor(
                batch_x, batch_y, max_seq_len)
            batches_x.append(tensor_batch_x)
            batches_y.append(tensor_batch_y)
            batch_x, batch_y = [seq], [label]
            max_seq_len = len(seq)
        else:
            batch_x.append(seq)
            batch_y.append(label)
            max_seq_len = max(max_seq_len, len(seq))

    tensor_batch_x, tensor_batch_y = pad_and_convert_to_tensor(
        batch_x, batch_y, max_seq_len)
    batches_x.append(tensor_batch_x)
    batches_y.append(tensor_batch_y)

    return batches_x, batches_y


def train(model, batches_x, batches_y, epochs, alpha):
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training with {model.pooling} pooling")
    start_time = time.time()

    for epoch in range(epochs):
        loss = torch.tensor(0.0)
        for batch_x, batch_y in zip(batches_x, batches_y):
            optimizer.zero_grad()
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            loss.backward()
            optimizer.step()
        print(
            f'Epoch {epoch + 1} Loss: {loss.item():.2f} - Learning rate: {scheduler.get_last_lr()[0]:.6f}')
        scheduler.step()

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

    print(
        f'Accuracy of the {model.pooling} pooling model: {correct / total * 100:.2f}%')


# -------------------------------verbose functions-------------------------------#

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


def batching_info(x_train_batches):
    print(f"\nNumber of training batches: {len(x_train_batches)}")

    memory_usages = get_memory_and_tokens_per_batch(x_train_batches)
    min_tokens = min(value[1] for value in memory_usages.values())
    max_tokens = max(value[1] for value in memory_usages.values())
    
    print(f"Smallest batch: {min_tokens} tokens")
    print(f"Largest batch: {max_tokens} tokens")
    print(f"Memory deviation between batches: {get_memory_usage_deviation(memory_usages):.2f}%")
