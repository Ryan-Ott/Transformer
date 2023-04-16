#!/usr/bin/env python3

import data_rnn
import fire
import utils

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 0.01
EMBEDDING_DIM = 128


def main(batch_by, epochs=EPOCHS, lr=LEARNING_RATE, emb_dim=EMBEDDING_DIM):
    """Load the IMDb dataset, train a model with max pooling and other with average pooling, and evaluate them.

    Args:
        batch_by (str): Method for batching the data. Can be 'instances' or 'tokens'.
        epochs (int, optional): Number of iterations during training. Defaults to 10.
        lr (float, o`ptional): Step size for training. Defaults to 0.01.
        emb_dim (int, optional): Dimensions of the embedding vector. Defaults to 128.
    """
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=False)
    PAD = w2i['.pad']  # Index of padding token (its 0)

    # Sort the training data by length
    x_train, y_train = zip(
        *sorted(zip(x_train, y_train), key=lambda x: len(x[0])))

    # Create batches
    if batch_by == 'instances':
        x_train_batches, y_train_batches = utils.batch_by_instances(x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = utils.batch_by_instances(x_val, y_val, pad_token=PAD)
    elif batch_by == 'tokens':
        x_train_batches, y_train_batches = utils.batch_by_tokens(x_train, y_train, pad_token=PAD)
        x_val_batches, y_val_batches = utils.batch_by_tokens(x_val, y_val)
    else:
        raise ValueError("batch_by must be set to 'instances' or 'tokens'")
    

    # Print the memory usage and number of tokens for each batch (useful when debugging batch_by_tokens)
    # memory_usages = utils.get_memory_usage_and_token_count_for_batches(x_train_batches, y_train_batches)
    # for i, memory_usage in enumerate(memory_usages):
        # print(f"Batch {i}: {memory_usage[0]} bytes | {memory_usage[1]} tokens")


    # print the hyperparameters
    print(f"\nHyperparameters: \nAlpha: {lr}\nEpochs: {epochs}")
    print("------------------------------------")

    # Create instances of the models
    model_max = utils.Model(len(i2w), n_classes, emb_dim, pooling='max')
    model_avg = utils.Model(len(i2w), n_classes, emb_dim, pooling='avg')

    # Train and evaluate the max pool model
    utils.train(model_max, x_train_batches, y_train_batches, epochs, lr)
    utils.evaluate(model_max, x_val_batches, y_val_batches)
    
    # Train and evaluate the avg pool model
    utils.train(model_avg, x_train_batches, y_train_batches, epochs, lr)
    utils.evaluate(model_avg, x_val_batches, y_val_batches)

    # Visualize the weights
    # weight_matrix_max = model_max.linear.weight.data.numpy()
    # weight_matrix_avg = model_avg.linear.weight.data.numpy()
    # utils.visualize_weights(weight_matrix_max)
    # utils.visualize_weights(weight_matrix_avg)

    # Visualize the embeddings
    embedding_matrix_max = model_max.embedding.weight.data.numpy()
    embedding_matrix_avg = model_avg.embedding.weight.data.numpy()
    utils.visualize_embeddings(embedding_matrix_max)
    utils.visualize_embeddings(embedding_matrix_avg)


if __name__ == '__main__':
    fire.Fire(main)
