import random
import numpy as np


class Network():
    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes

        #Inicializa os pesos do bias de todas as conexões aleatoriamente
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        #Conecta todos os perceptrons com todos os outros da camada anterior
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
