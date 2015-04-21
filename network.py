import numpy as np


class Network():
    def __init__(self, sizes):
        self.num_layer = len(sizes)
        self.sizes = sizes

        # Conecta todos os perceptrons com todos os outros da camada anterior e adiciona os bias
        self.weights = [np.random.randn(y, x+1) * 0.25 for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, input, result, lrate, iterations):
        for i in range(iterations):
            # Escolhe um valor de entrada aleatorio
            k = np.random.randint(len(input))

            data = input[k]
            target = result[k][0]

            # Cria uma matriz do tamanho dos dados com uma linha a mais.
            temp = np.ones([data.shape[0] + 1, data.shape[1]])
            # Copia os valores do input para todas as linhas menos a ultima que e o bias.
            temp[0:-1, :] = data
            # Copia de volta para o input essa nova matriz com o bias adicionado
            data = temp.T

            # Adiciona o valor de entrada como primeiro valor da ativacao
            a = [data]

            # Para cada camada, multiplica os pesos pelas entradas e adiciona
            # no vetor "a" que representa a ativacao e entradas para a proxima camada.
            for layer in range(len(self.weights)):
                temp = []
                for p in range(len(self.weights[layer])):
                    temp.append(self.sigmoid(np.dot(a[layer][0], self.weights[layer][p])))
                if layer < self.num_layer-2:
                    temp.append(1)
                tempa = np.array([temp])
                a.append(tempa)

            # Compara saida esperada com a saida dessa entrada aleatoria
            output = a[-1][0]
            if(i % 100 == 0):
                print(i)

            error = target - output

            # Calcula os erros de cada perceptron da camada de saida
            deltas = [error]

            # Comecando da penultima camada calculamos os erros
            # das camadas anteriores já calculando com a derivada da ativacao
            for layer in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[layer]))

            # Inverte os erros, comecando agora da primeira camada
            deltas.reverse()

            # Passa por cada perceptron de cada camada arrumando
            # todos os pesos das conexoes ligadas a ele
            for layer in range(len(self.weights)):
                # Pre carrega os inputs do layer para otimazar loop
                linput = a[layer][0]
                for perceptron in range(len(self.weights[layer])):
                    # Pre calcula a differenca do perceptron para otimizar loop
                    diff = lrate * deltas[layer][perceptron] * self.sigmoid_prime(a[layer+1][0][perceptron])
                    for connection in range(len(self.weights[layer][perceptron])):
                        #Aplica a derivada da ativação e a entrada da camada nessa diferenca
                        change = diff * linput[connection]
                        self.weights[layer][perceptron][connection] += change

    def process(self, x):

        # Cria uma matriz do tamanho dos dados com uma linha a mais.
        temp = np.ones([x.shape[0] + 1, x.shape[1]])
        # Copia os valores do input para todas as linhas menos a ultima que e o bias.
        temp[0:-1, :] = x
        # Copia de volta para o input essa nova matriz com o bias adicionado
        x = temp.T

        # Adiciona o valor de entrada como primeiro valor da ativacao
        a = [x]

        # Segue o mesmo principio do aprendizado, dessa vez memorizando apenas
        # a saida a ultima camada.
        for layer in range(len(self.weights)):
            temp = []
            for p in range(len(self.weights[layer])):
                temp.append(self.sigmoid(np.dot(a[0], self.weights[layer][p])))
            if layer < self.num_layer-2:
                temp.append(1)
            tempa = np.array([temp])
            a = tempa

        return a


    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_prime(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
