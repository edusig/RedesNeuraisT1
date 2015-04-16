import numpy as np


class Network():
    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes

        #Inicializa os pesos do bias de todas as conexoes aleatoriamente
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        #Conecta todos os perceptrons com todos os outros da camada anterior
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        print(self.biases)
        for layer in self.weights:
            for p in layer:
                print(p)
            print

    def feedforward(self, inputs, weights, bias):
        activation = self.sigmoid(sum(np.dot(inputs, weights)) + bias)
        return activation

    def backpropagation(self, deltas, lrate, data):
        errors = [deltas]
        for l in range(1, self.num_layer-1):
            newdeltas = []
            for p in range(len(self.weights[l])):
                newdeltas.append(sum([x * y for x, y in zip(self.weights[-l][p], deltas)]))
            errors.append(newdeltas)
        errors.reverse()
        for l in range(1, self.num_layer):
            for p in range(len(self.weights[l])):
                for w in range(len(self.weights[l][p])):
                    self.weights[l][p][w] = self.weights[l][p][w] + lrate * (deltas[l][p] * sigmoid_prime(data[l][p]) * data[l-1][p])
        return

    def gradient(self, input, lrate, epochs):
        target = input[1]
        for e in range(epochs):
            data = input[0]
            for layer in range(1, self.num_layer-1):
                a = []
                for p in range(len(self.weights[layer])):
                    a.append(self.feedforward(data[layer-1], self.weights[layer-1][p], self.bias[layer-1][p]))
                data.append(a)
            #mse = sum([pow(x - y) for x, y in zip(data[-1], target)]) / 2
            deltas = [x - y for x, y in zip(data[-1], target)]
            self.backpropagation(deltas, lrate, data)

        return

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
