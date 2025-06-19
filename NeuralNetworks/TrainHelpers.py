import numpy as np

def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def accuracy(probs, labels):
    return np.mean(np.argmax(probs, axis=1) == labels)

def recall(probs, labels):
    return np.mean(np.logical_and(np.argmax(probs, axis=1) == labels, labels == 1))

def precision(probs, labels):
    return np.mean(np.logical_and(np.argmax(probs, axis=1) == labels, labels == 1))

def f1(probs, labels):
    return 2 * precision(probs, labels) * recall(probs, labels) / (precision(probs, labels) + recall(probs, labels))

def false_positives(probs, labels):
    return np.mean(np.logical_and(np.argmax(probs, axis=1) == labels, labels == 0))


def get_stats(model, labels):