import gzip
import math
import os
import pickle
import random
import tarfile
import time
from typing import List, Tuple
import wget

import numpy as np
from tokenizers import Tokenizer
import torch
import torch.distributions as dist
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn


def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):
    cst = 'char' if char else 'word'

    imdb_url = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'.format(cst)
    imdb_file = 'imdb.{}.pkl.gz'.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set(random.sample(range(len(sequences['train'])), k=val))
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


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


def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)  # categorical distribution

    return cd.sample()


def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.
    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    """
    with gzip.open(path) if path.endswith('.gz') else open(path, 'rb') as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)


def sample_batch(data, length, batch_size):  # TODO: read more into this
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # Sample the starting indices of the sequences to slice out.
    starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # Slice out the input sequences
    seqs_inputs  = [data[start:start + length] for start in starts]
    # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    #    next character at each position)

    # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    return inputs, target


def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):  # TODO: read more into this
    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.

    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True) # type: ignore # chr() expects an int, not a tensor

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed


def preprocess(x: List[str], y: List[str], tokenizer: Tokenizer, batch_size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create batches of tokenized and padded documents and summaries.

        Args:
            x (List[str]): List of documents.
            y (List[str]): List of summaries.
            tokenizer (Tokenizer): Tokenizer to use.
            batch_size (int): Batch size.
            device (torch.device): Device to save tensors on.
            
        Returns:
            List[Tuple[torch.Tensor, torch.Tensor]]: List of batches of tokenized and padded documents and summaries.
        """
        batches = []
        for i in range(0, len(x), batch_size):
            # Get the batch
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            # Tokenize the batch
            tokenizer.enable_padding(pad_id=tokenizer.token_to_id('[PAD]'), pad_token='[PAD]')
            x_batch = [i.ids for i in tokenizer.encode_batch(x_batch)]
            y_batch = [i.ids for i in tokenizer.encode_batch(y_batch)]

            # Convert to tensors and save on device
            x_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
            y_batch = torch.tensor(y_batch, dtype=torch.long, device=device)

            batches.append((x_batch, y_batch))

        return batches


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


def batch_by_tokens(device, sequences, labels, max_tokens=2**15, pad_token=0):
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
