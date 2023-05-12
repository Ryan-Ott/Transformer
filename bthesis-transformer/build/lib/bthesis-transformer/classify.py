#!/usr/bin/env python3

import data_rnn
import fire
import models
import torch
import utils

# Hyperparameters
EPOCHS = 3
LEARNING_RATE = 0.001
EMBEDDING_DIM = 256


def main(model, b="tokens", e=EPOCHS, a=LEARNING_RATE, k=EMBEDDING_DIM, p="avg", h=4, d=4, f=False, v=False):
    """Load the IMDb dataset, train a classification model and evaluate the accuracy.

    Args:
        model (str): Model to use. Can be 'base', 'simple', 'multi' or 'full'.
        b (str, optional): batch_by - Method for batching the data. Can be 'instances' or 'tokens'. Defaults to 'tokens'.
        e (int, optional): Number of iterations during training. Defaults to 3.
        a (float, optional): Step size for training. Defaults to 0.001.
        k (int, optional): Dimensions of the embedding vector. Defaults to 512.
        p (str, optional): Pooling method. Can be 'avg' or 'max'. Defaults to 'avg'.
        h (int, optional): Number of heads in the multi-head attention layer. Defaults to 4.
        d (str, optional): Depth of the transformer, so number of blocks. Defaults to 4.
        f (bool, optional): Use the final (testing) dataset. Defaults to False.
        v (bool, optional): Verbose. Prints batch information and produces graphs. Defaults to False.
    """
    batch_by, epochs, alpha, emb_dim, pool, heads, depth, final = b, e, a, k, p, h, d, f
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=final)
    PAD = w2i['.pad']  # Index of padding token (its 0)

    # Sort the training data by length (shortest to longest)
    x_train, y_train = utils.sort(x_train, y_train)

    # Get the maximum sequence length as the last element of the sorted list
    max_len = len(x_train[-1])

    # Create batches
    x_train_batches, y_train_batches, x_val_batches, y_val_batches = utils.batchify(device, batch_by, x_train, y_train, x_val, y_val, PAD)

    # Create instances of the models
    name = model
    if model == "base":
        model = models.BaseClf(len(i2w), n_classes, emb_dim, pool)
    elif model == "simple":
        model = models.SimpleClf(len(i2w), n_classes, emb_dim, pool)
    elif model == "multi":
        model = models.MultiheadClf(len(i2w), n_classes, emb_dim, pool, heads)
    elif model == "full":
        model = models.ClfTransformer(len(i2w), emb_dim, n_classes, max_len, heads, depth)
    else:
        raise ValueError("model must be set to 'base', 'simple' or 'multi'")
    model.to(device)

    # print the hyperparameters
    print(f"\nModel: {name}\nEpochs: {epochs}\nAlpha: {alpha}\nEmbedding dimension: {emb_dim}\nHeads: {heads}\nPool: {model.pooling}\nBatch by: {batch_by}\nDevice: {device}")
    print("------------------------------------------------------------------------")
    
    # Train and evaluate the model
    utils.train(model, x_train_batches, y_train_batches, epochs, alpha)
    utils.evaluate(model, x_val_batches, y_val_batches)

    if v:
        utils.batching_info(x_train_batches)
        # TODO: produce graphs


if __name__ == '__main__':
    fire.Fire(main)
    # main(model="base", b="tokens", e=1, a=0.001, k=256, h=4, v=False)  # for debugging
