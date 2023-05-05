#!/usr/bin/env python3

import base
import simple
import multihead
import data_rnn
import fire
import utils

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256


def main(model, bb="tokens", epochs=EPOCHS, lr=LEARNING_RATE, emb_dim=EMBEDDING_DIM, pool="avg", heads=4, v=False):
    """Load the IMDb dataset, train a classification model and evaluate the accuracy.

    Args:
        model (str): Model to use. Can be 'base', 'simple' or 'multihead'.
        bb (str, optional): batch_by - Method for batching the data. Can be 'instances' or 'tokens'. Defaults to 'tokens'.
        epochs (int, optional): Number of iterations during training. Defaults to 3.
        lr (float, optional): Step size for training. Defaults to 0.001.
        emb_dim (int, optional): Dimensions of the embedding vector. Defaults to 512.
        pool (str, optional): Pooling method. Can be 'avg' or 'max'. Defaults to 'avg'.
        heads (int, optional): Number of heads in the multi-head attention layer. Defaults to 4.
        v (bool, optional): Verbose. Defaults to False.
    """
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=False)
    PAD = w2i['.pad']  # Index of padding token (its 0)

    # Sort the training data by length (shortest to longest)
    x_train, y_train = sort(x_train, y_train)

    # Create batches
    x_train_batches, y_train_batches, x_val_batches, y_val_batches = batchify(bb, x_train, y_train, x_val, y_val, PAD)

    # Create instances of the models
    name = model
    if model == "base":
        model = base.Transformer(len(i2w), n_classes, emb_dim, pool)
    elif model == "simple":
        model = simple.Transformer(len(i2w), n_classes, emb_dim, pool)
    elif model == "multi":
        model = multihead.Transformer(len(i2w), n_classes, emb_dim, pool, heads)
    else:
        raise ValueError("model must be set to 'base', 'simple' or 'multi'")

    # print the hyperparameters
    print(f"\nModel: {name}\nEpochs: {epochs}\nAlpha: {lr}\nEmbedding dimension: {emb_dim}\nHeads: {heads}\nPool: {model.pooling}\nBatch by: {bb}")
    print("------------------------------------------------------------------------")
    
    # Train and evaluate the model
    utils.train(model, x_train_batches, y_train_batches, epochs, lr)
    utils.evaluate(model, x_val_batches, y_val_batches)

    if v:
        if bb=="tokens":  # Print the memory usage and number of tokens for each batch (useful when debugging batch_by_tokens)
            memory_usages = utils.get_memory_and_tokens_per_batch(x_train_batches)
            min_tokens = min(value[1] for value in memory_usages.values())
            max_tokens = max(value[1] for value in memory_usages.values())
            print(f"Smallest batch: {min_tokens} tokens")
            print(f"Largest batch: {max_tokens} tokens")
            print(f"Memory deviation between batches: {utils.get_memory_usage_deviation(memory_usages):.2f}%")

        # Visualize the weights
        weight_matrix = model.linear.weight.data.numpy()
        utils.visualize_weights(weight_matrix, title="Avg Pooling Weights")

        # Visualize the embeddings
        embedding_matrix = model.embedding.weight.data.numpy()
        utils.visualize_embeddings(embedding_matrix, title="Avg Pooling Embeddings")

def sort(x, y):
    x, y = zip(*sorted(zip(x, y), key=lambda x: len(x[0])))
    return x, y


def batchify(batch_by, x_train, y_train, x_val, y_val, PAD):
    """Create batches of the training and validation data."""
    if batch_by == 'instances':
        x_train_batches, y_train_batches = utils.batch_by_instances(x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = utils.batch_by_instances(x_val, y_val, pad_token=PAD)
    elif batch_by == 'tokens':
        x_train_batches, y_train_batches = utils.batch_by_tokens(x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = utils.batch_by_tokens(x_val, y_val, pad_token=PAD)
    else:
        raise ValueError("batch_by must be set to 'instances' or 'tokens'")
    
    return x_train_batches, y_train_batches, x_val_batches, y_val_batches


if __name__ == '__main__':
    fire.Fire(main)
    # main(model="base", bb="tokens", epochs=1, lr=0.001, emb_dim=256, heads=4, v=False)
