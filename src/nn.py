import numpy as np


def sigmoid(X):
    return 1/(1+np.exp(-X))


class NeuralNetwork:

    # neuroN - array with n of neurons in each layer (at least 2 layers: in, out)
    def __init__(self, neuronN, network=None):

        if network == None:
            self.layersN = len(neuronN)
            self.weighs = []
            self.biases = []

            for i in range(1, self.layersN):
                self.weighs.append(np.random.rand(neuronN[i-1], neuronN[i]))
                self.biases.append(2 * np.random.rand(neuronN[i], 1) - 1)

        else:
            self.layersN = network.layersN
            self.weighs = []
            self.biases = []
            for weigh in network.weighs:
                self.weighs.append(weigh.copy())
            for bias in network.biases:
                self.biases.append(bias.copy())

    def predict(self, input):

        output = np.matmul(input, self.weighs[0])
        output = np.add(output, self.biases[0])
        output = list(map(sigmoid, output))

        for i in range(1, self.layersN - 1):
            output = np.matmul(output, self.weighs[i])
            output = np.add(output, self.biases[i-1])
            output = list(map(sigmoid, output))

        return output

    def mutate(self, chance):

        for w in self.weighs:
            for x in w:
                if np.random.rand() < chance:
                    randomNumber = np.random.normal(0, 0.1)
                    x += randomNumber

        for b in self.biases:
            for x in b:
                if np.random.rand() < chance:
                    randomNumber = np.random.normal(0, 0.1)
                    b += randomNumber

    def saveWeighs(self):

        for i in range(0, self.layersN - 1):
            np.savetxt('weighs' + str(i) + '.csv',
                       self.weighs[i], delimiter=',')
            np.savetxt('biases' + str(i) + '.csv',
                       self.biases[i], delimiter=',')

    def readWeighs(self):

        self.weighs = []
        self.biases = []
        for i in range(0, self.layersN - 1):
            self.weighs.append(np.loadtxt(
                'weighs' + str(i) + '.csv', delimiter=','))
            self.biases.append(np.loadtxt(
                'biases' + str(i) + '.csv', delimiter=','))
