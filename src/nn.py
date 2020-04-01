import numpy as np


def sigmoid(X):
    return 1/(1+np.exp(-X))


class NeuralNetwork:

    def __init__(self, in_nodes, h_nodes, out_nodes, network=None):

        if network == None:
            self.input_nodesN = in_nodes
            self.hidden_nodesN = h_nodes
            self.output_nodesN = out_nodes
            self.weighs_ih = np.random.rand(in_nodes, h_nodes)
            self.weighs_ho = np.random.rand(h_nodes, out_nodes)
            self.bias_h = 2 * np.random.rand(h_nodes, 1) - 1
            self.bias_o = 2 * np.random.rand(out_nodes, 1) - 1
        else:
            self.input_nodesN = network.input_nodesN
            self.hidden_nodesN = network.hidden_nodesN
            self.output_nodesN = network.output_nodesN
            self.weighs_ih = network.weighs_ih.copy()
            self.weighs_ho = network.weighs_ho.copy()
            self.bias_h = network.bias_h.copy()
            self.bias_o = network.bias_o.copy()

    def predict(self, input):

        h_neurons = np.matmul(input, self.weighs_ih)
        h_neurons = np.add(h_neurons, self.bias_h)
        h_neurons_m = map(sigmoid, h_neurons)

        output = np.matmul(list(h_neurons_m), self.weighs_ho)
        output = np.add(output, self.bias_o)
        output_m = map(sigmoid, output)

        return list(output_m)

    def mutate(self, chance):

        for weigh in self.weighs_ih:
            if np.random.rand() < chance:
                #randomNumber = 2 * np.random.rand() - 1
                randomNumber = np.random.normal(0, 0.1)
                weigh += randomNumber

        for weigh in self.weighs_ho:
            if np.random.rand() < chance:
                #randomNumber = 2 * np.random.rand() - 1
                randomNumber = np.random.normal(0, 0.1)
                weigh += randomNumber

        for bias in self.bias_h:
            if np.random.rand() < chance:
                #randomNumber = 2 * np.random.rand() - 1
                randomNumber = np.random.normal(0, 0.1)
                bias += randomNumber

        for bias in self.bias_o:
            if np.random.rand() < chance:
                #randomNumber = 2 * np.random.rand() - 1
                randomNumber = np.random.normal(0, 0.1)
                bias += randomNumber

    def saveWeighs(self):

        savetxt('weighs_ih.csv', self.weighs_ih, delimiter=',')
        savetxt('weighs_ho.csv', self.weighs_ho, delimiter=',')
