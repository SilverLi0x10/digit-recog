import random, time
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_prime(a, y):
    return a - y


class Network:
    def __init__(self, sizes):
        self.layer_num = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward_az(self, a, ws=None, bs=None):
        if ws is None:
            ws = self.weights
        if bs is None:
            bs = self.biases

        acs = [a]
        zs = []
        for w, b in zip(ws, bs):
            z = w @ a + b
            zs.append(z)
            a = sigmoid(z)
            acs.append(a)
        return acs, zs

    def feed_forward(self, a, ws=None, bs=None):
        return self.feed_forward_az(a, ws, bs)[0][-1]

    def evaluate(self, test_data):
        return sum(np.argmax(self.feed_forward(x)) == y for x, y in test_data)

    def cost(self, x, y, ws=None, bs=None):
        a = self.feed_forward(x, ws, bs)
        n = y.shape[0]
        return np.sum((y - a) ** 2) / (2 * n)

    def cost_w(self, x, y, ws):
        return self.cost(x, y, ws=ws)

    def cost_b(self, x, y, bs):
        return self.cost(x, y, bs=bs)

    def numerical_gradient(self, x, y, epsilon=1e-5):
        cost = lambda: self.cost(x, y)
        C = cost()

        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        for l, w in enumerate(self.weights):
            it = np.nditer(w, flags=["multi_index"], op_flags=[["readwrite"]])
            while not it.finished:
                idx = it.multi_index
                old_val = w[idx]
                w[idx] = old_val + epsilon
                nabla_w[l][idx] = (cost() - C) / epsilon
                w[idx] = old_val
                it.iternext()

        for l, b in enumerate(self.biases):
            it = np.nditer(b, flags=["multi_index"], op_flags=[["readwrite"]])
            while not it.finished:
                idx = it.multi_index
                old_val = b[idx]
                b[idx] = old_val + epsilon
                nabla_b[l][idx] = (cost() - C) / epsilon
                b[idx] = old_val
                it.iternext()

        return nabla_w, nabla_b

    def update_batch(self, batch, eta):
        nabla_w = [np.zeros_like(w) for w in self.weights]
        nabla_b = [np.zeros_like(b) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.numerical_gradient(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        size = len(batch)
        self.weights = [w - eta * (nw / size) for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * (nb / size) for b, nb in zip(self.biases, nabla_b)]

    # shuffle gradient down
    def train(self, train_data, epochs, batch_size, eta, test_data=None):
        n = len(train_data)
        st_time = time.process_time()

        for i in range(epochs):
            random.shuffle(train_data)
            batches = [train_data[j : j + batch_size] for j in range(0, n, batch_size)]

            for batch in batches:
                self.update_batch(batch, eta)

            if test_data:
                print(
                    f"Epoch {i}, time: {time.process_time()-st_time}s, {self.evaluate(test_data)} / {len(test_data)}"
                )
            else:
                print(f"Epoch {i}, time: {time.process_time()-st_time}s")


"""
Usage:

import mnist_loader as ml
tr,v,te=ml.load_data_wrapper()
import network as nw
net = nw.Network([784, 30, 10])
net.train(tr, 100, 10, 3, te)
"""
