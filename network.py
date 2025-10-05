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

    def feed_forward_az(self, a):
        acs = [a]
        zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            acs.append(a)
        return acs, zs

    def feed_forward(self, a):
        return self.feed_forward_az(a)[0][-1]

    def evaluate(self, test_data):
        return sum(np.argmax(self.feed_forward(x)) == y for x, y in test_data)

    def back_propagation(self, x, y):
        acs, zs = self.feed_forward_az(x)

        delta = cost_prime(acs[-1], y) * sigmoid_prime(zs[-1])
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        def update(l):
            nabla_w[-l] = np.dot(delta, acs[-l - 1].transpose())
            nabla_b[-l] = delta

        update(1)

        for l in range(2, self.layer_num):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(
                zs[-l]
            )
            update(l)

        return nabla_w, nabla_b

    def update_batch(self, batch, eta):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        for x, y in batch:
            delta_nabla_w, delta_nabla_b = self.back_propagation(x, y)
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
