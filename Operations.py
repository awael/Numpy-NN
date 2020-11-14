import math
import numpy as np


def RELU(s):
    # if s >= 0:
    #     return s
    # else:
    #     return 0.01 * s
    return s * (s > 0)


def RELU_derv(s):
    return 1 * (s > 0)
    # if s >= 0:
    #     return 1
    # else:
    #     return 0.01


def tanh(s):
    return (2. / (1. + np.exp(-2 * s))) - 1


def tanh_derv(s):
    ftanh = (2. / (1. + np.exp(-2 * s))) - 1
    return 1 - ftanh * ftanh


def sigmoid(s):
    return 1 / (1 + np.exp(-s))


def sigmoid_derv(s):
    return s * (1 - s)


def softmax(s):
    exps = np.exp(s - np.max(s, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res / n_samples


def error(pred, real):
    n_samples = real.shape[0]
    logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
    loss = np.sum(logp) / n_samples
    return loss
