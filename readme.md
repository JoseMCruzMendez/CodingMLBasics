# Coding The Basics of Machine Learning

This repo is meant to document my progress in exploring different concepts in ML, from optimizers to modern architectures, in a step-by-step fashion. 

I have some previous experience implementing some basic models (Naive Bayes Classifiers, Decision Trees/Random Forests, MLPs) for a college class, and thought it would be a good idea to explore further and implement more things from scratch to understand their inner workings. As such, I will skip straight to MLPs for implementation.

The rules are simple: I can only "abstract" away implementations for more stable ones in libraries like pytorch once I have implemented them myself. This means I will start the optimizers section by coding my own grad-graph, and will not use pytorch optimizers until I have coded ADAM on my own.

**Status: in progress.** Checked items are implemented and tested; the rest are
still ahead of me. Graph neural networks are where I stopped.

- Optimizers:
  - [x] SGD
  - [x] Momentum
  - [x] Nesterov Momentum
  - [x] AdaGrad
  - [x] RMSProp
  - [x] Adam
  - [x] Decoupled weight decay (AdamW, RMSPropW, AdaGradW, MomentumSGDW)
- Models:
  - [x] Autoencoders
  - [ ] Variational autoencoders
  - [x] RNNs
  - [x] LSTM
  - [x] GRU
  - [x] ExpRNN / expGRU (orthogonal parametrization with ModReLU)
  - [ ] Liquid Neural Networks
  - [x] CNN
  - [x] Transformers
  - [ ] XGBoost
  - [ ] Graph Neural Networks — next up

A few things that aren't obvious from the file names. `Convnets.py` implements
convolution and max-pooling with hand-written forward *and* backward passes.
The optimizers are an inheritance chain, so each one is visibly a modification
of the one before it rather than a separate implementation. And `ExpRNN` /
`expGRU` use an exponential-map parametrization with `ModRelu` to keep the
recurrent weights orthogonal. That one came out of the cheap-orthogonal-
constraints work (Mostly by Lezcano-Casado), and it changed how I think about parametrizing weights more
than anything else here.

Each section has a learning log under `LearningLogs/`, written as a Kolb cycle:
what I ran, what surprised me, what principle it recovered, what I would try
next. Those record the bugs and wrong turns rather than the cleaned-up result.
