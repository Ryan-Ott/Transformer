# Transformers for NLP Bachelor Project

This repository contains all the files for the bachelor project on Transformers for Natural Language Processing at the Vrije Universiteit Amsterdam by Ryan Ott.

### Updates

**Week 1** - Creating a simple sentiment classification model consisting of an embedding layer and a global pooling operation.

The challenge of this week is to get the batching up and running from scratch and testing which global pooling method achieves the highest performance in terms of accuracy.

**Week 2** - Added linear layer to classify into two classes. Played around with hyperparameter optimization.

Next to that, first steps for the attention implementation were taken in preparation for the next weeks.

**Weeks 3 & 4** - Three models: base, simple and multi were created and tested.

Base model is only the token embedding + pooling + linear projection.
Simple model incorporates simple self-attention.
Multi employs multi-head self-attention.

**Week 5** - Full transformer classification model.

Builds on top of multi, but now has a dedicated encoder block that is repeated *d* times. Includes further improvements like learning rate scheduling. Hyperparameter tuning still needed.

**Week 6** - Moving onto making a generative transformer model.