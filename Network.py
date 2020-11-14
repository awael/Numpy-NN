from Operations import *
import math
import numpy as np


class Network:
    def __init__(self, x, y, n, hlayers):
        self.x = x
        self.neurons = n
        neurons = n
        self.lr = 0.02
        self.alpha = 0.6
        ip_dim = x.shape[1]
        op_dim = y.shape[1]

        # Xavier random initialization
        np.random.seed(0)
        scale = 1 / max(1., (ip_dim + op_dim) / 2.)
        limit = math.sqrt(3.0 * scale)

        self.layers = hlayers + 2
        self.w = []
        self.b = []
        self.vw = []
        self.vb = []

        self.w.append(np.random.uniform(-limit, limit, size=(ip_dim, neurons)))
        self.b.append(np.random.uniform(-limit, limit, size=(1, neurons)))

        for layer in range(hlayers):
            self.w.append(np.random.uniform(-limit, limit, size=(neurons, neurons)))
            self.b.append(np.random.uniform(-limit, limit, size=(1, neurons)))

        self.w.append(np.random.uniform(-limit, limit, size=(neurons, op_dim)))
        self.b.append(np.random.uniform(-limit, limit, size=(1, op_dim)))

        # Nesterov Momentum variables initialization
        self.vw.append(np.random.uniform(-limit, limit, size=(ip_dim, neurons)))
        self.vb.append(np.random.uniform(-limit, limit, size=(1, neurons)))

        for layer in range(hlayers):
            self.vw.append(np.random.uniform(-limit, limit, size=(neurons, neurons)))
            self.vb.append(np.random.uniform(-limit, limit, size=(1, neurons)))

        self.vw.append(np.random.uniform(-limit, limit, size=(neurons, op_dim)))
        self.vb.append(np.random.uniform(-limit, limit, size=(1, op_dim)))

        self.y = y
        self.errors = []

    def feedforward(self):
        # Nesterov Momentum variables
        W_aheadw = []
        W_aheadb = []
        z = []
        self.a = []

        for i in range(self.layers):
            W_aheadw.append(self.vw[i] * self.alpha + self.w[i])
            W_aheadb.append(self.vb[i] * self.alpha + self.b[i])

        z.append(np.dot(self.x, W_aheadw[0]) + W_aheadb[0])
        self.a.append(tanh(z[0]))

        for i in range(self.layers - 2):
            z.append(np.dot(self.a[i], W_aheadw[i + 1]) + W_aheadb[i + 1])
            self.a.append(tanh(z[i + 1]))

        z.append(np.dot(self.a[-1], W_aheadw[self.layers - 1]) + W_aheadb[self.layers - 1])
        self.a.append(softmax(z[self.layers - 1]))

    def backprop(self):
        loss = error(self.a[-1], self.y)
        # print('Error :', loss)
        self.errors.append(loss)
        a_delta = []
        z_delta = []

        a_delta.append(cross_entropy(self.a[-1], self.y))
        z_delta.append(np.dot(a_delta[-1], self.w[-1].T))
        for i in range((self.layers - 2), 0, -1):
            a_delta.insert(0, (z_delta[0] * tanh_derv(self.a[i])))  # w hidden
            z_delta.insert(0, (np.dot(a_delta[0], self.w[i].T)))

        a_delta.insert(0, (z_delta[0] * tanh_derv(self.a[0])))  # w input

        # Nesterov Momentum variables
        self.vw[-1] = self.alpha * self.vw[-1] - (self.lr * np.dot(self.a[-2].T, a_delta[-1]))
        self.vb[-1] = self.alpha * self.vb[-1] - (self.lr * np.sum(a_delta[-1], axis=0, keepdims=True))

        for i in range((self.layers - 2), 0, -1):
            if a_delta[i].ndim == 1:
                a_delta[i] = np.reshape(a_delta[i], (np.shape(a_delta[i])[0], 1))

            self.vw[i] = self.alpha * self.vw[i] - (self.lr * np.dot(self.a[i - 1].T, a_delta[i]))
            self.vb[i] = self.alpha * self.vb[i] - (self.lr * np.sum(a_delta[i], axis=0))

        self.vw[0] = self.alpha * self.vw[0] - (self.lr * np.dot(self.x.T, a_delta[0]))
        self.vb[0] = self.alpha * self.vb[0] - (self.lr * np.sum(a_delta[0], axis=0))

        # Update weights
        for i in range(self.layers):
            self.w[i] += self.vw[i]
            self.b[i] += self.vb[i]

    def predict(self, data):
        self.x = data
        self.feedforward()
        return self.a[-1].argmax()
