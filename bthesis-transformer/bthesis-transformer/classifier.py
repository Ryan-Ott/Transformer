#!/usr/bin/env python3

import data_rnn
import fire
import utils

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 0.001
EMBEDDING_DIM = 128


def main(bb, epochs=EPOCHS, lr=LEARNING_RATE, emb_dim=EMBEDDING_DIM, v=False):
    """Load the IMDb dataset, train a model with max pooling and other with average pooling, and evaluate them.

    Args:
        bb (str): batch_by - Method for batching the data. Can be 'instances' or 'tokens'.
        epochs (int, optional): Number of iterations during training. Defaults to 10.
        lr (float, o`ptional): Step size for training. Defaults to 0.01.
        emb_dim (int, optional): Dimensions of the embedding vector. Defaults to 128.
        v (bool, optional): Verbose. Defaults to False.
    """
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=False)
    PAD = w2i['.pad']  # Index of padding token (its 0)

    # Sort the training data by length
    x_train, y_train = zip(*sorted(zip(x_train, y_train), key=lambda x: len(x[0])))

    # Create batches
    x_train_batches, y_train_batches, x_val_batches, y_val_batches = batchify(bb, x_train, y_train, x_val, y_val, PAD)

    # print the hyperparameters
    print(f"\nHyperparameters: \nAlpha: {lr}\nEpochs: {epochs}")
    print("------------------------------------")

    # Create instances of the models
    model_max = utils.Transformer(len(i2w), n_classes, emb_dim, pooling='max')
    model_avg = utils.Transformer(len(i2w), n_classes, emb_dim, pooling='avg')

    # Train and evaluate the max pool model
    utils.train(model_max, x_train_batches, y_train_batches, epochs, lr)
    utils.evaluate(model_max, x_val_batches, y_val_batches)
    
    # Train and evaluate the avg pool model
    utils.train(model_avg, x_train_batches, y_train_batches, epochs, lr)
    utils.evaluate(model_avg, x_val_batches, y_val_batches)

    if v:
        if bb=="tokens":  # Print the memory usage and number of tokens for each batch (useful when debugging batch_by_tokens)
            memory_usages = utils.get_memory_and_tokens_per_batch(x_train_batches)
            min_tokens = min(value[1] for value in memory_usages.values())
            max_tokens = max(value[1] for value in memory_usages.values())
            print(f"Smallest batch: {min_tokens} tokens")
            print(f"Largest batch: {max_tokens} tokens")
            print(f"Memory deviation between batches: {utils.get_memory_usage_deviation(memory_usages):.2f}%")

        # Visualize the weights
        weight_matrix_max = model_max.linear.weight.data.numpy()
        weight_matrix_avg = model_avg.linear.weight.data.numpy()
        utils.visualize_weights(weight_matrix_max, title="Max Pooling Weights")
        utils.visualize_weights(weight_matrix_avg, title="Avg Pooling Weights")

        # Visualize the embeddings
        embedding_matrix_max = model_max.embedding.weight.data.numpy()
        embedding_matrix_avg = model_avg.embedding.weight.data.numpy()
        utils.visualize_embeddings(embedding_matrix_max, title="Max Pooling Embeddings")
        utils.visualize_embeddings(embedding_matrix_avg, title="Avg Pooling Embeddings")


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
