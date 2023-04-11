#!/usr/bin/env python3

import data_rnn
import utils

# Hyperparameters
ITERATIONS = 10
LEARNING_RATE = 0.01

if __name__ == '__main__':
    # Load the IMDb dataset
    (x_train, y_train), (x_val, y_val), (i2w, w2i), n_classes = data_rnn.load_imdb(final=False)
    PAD = w2i['.pad']  # Index of padding token (its 0)

    # Sort the training data by length
    x_train, y_train = zip(*sorted(zip(x_train, y_train), key=lambda x: len(x[0])))

    # Create batches
    x_train, y_train = utils.batch_by_instances(x_train, y_train)  # batch_by_tokens(x_train, y_train)
    x_val, y_val = utils.batch_by_instances(x_val, y_val)  # batch_by_tokens(x_val, y_val)

    # Create an instance of the max pool model
    model_max = utils.Model(len(i2w), pooling='max')

    # Train the max model
    utils.train(model_max, x_train, y_train, ITERATIONS, LEARNING_RATE)

    # Evaluate the model
    utils.evaluate(model_max, x_val, y_val)

    # Do the same for the average pooling model
    model_avg = utils.Model(len(i2w), pooling='avg')
    utils.train(model_avg, x_train, y_train, ITERATIONS, LEARNING_RATE)
    utils.evaluate(model_avg, x_val, y_val)
