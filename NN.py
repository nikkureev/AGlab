import numpy as np


class Neuron():

    def __init__(self, weights):
        self.weights = weights

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        return x * (x - 1)

    def multiplying(self, var_a, var_b):
        summary = 0
        if type(var_a) == list and type(var_a) == list:
            for i, j in zip(var_a, var_b):
                summary += i * j
        elif type(var_a) == int:
            summary += var_a * var_b[0]
        elif type(var_b) == int:
            summary += var_a[0] * var_b
        return summary

    def activation(self, inputs):
        self.inputs = inputs
        self.dot_result = self.multiplying(inputs, self.weights)
        self.outputs = self.sigmoid(self.dot_result)
        return self.outputs


class Layer():

    def __init__(self, number_of_inputs, number_of_outputs):
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs

        self.weights = 2 * np.random.random((self.number_of_outputs, self.number_of_inputs)) - 1

        self.neurons = []
        for i in range(self.number_of_outputs):
            self.neurons.append(Neuron(2 * np.random.random((self.number_of_inputs)) - 1))


class NeuralNetwork():

    def __init__(self, number_of_layers):
        self.layers = []
        number_of_layers.insert(0, 1)
        for i in range(len((number_of_layers)) - 1):
            self.layers.append(Layer(number_of_layers[i], number_of_layers[i + 1]))

    def for_adjustments(self, var_list, var_int):
        if type(var_list) == list:
            assert type(var_int) == np.float64
            #assert type(var_list) == list
            out_list = []
            for i in var_list:
                out_list.append(i * var_int)
        else:
            return var_list * var_int
        return out_list

    def feedforward(self, inputs):
        first = True
        data = inputs
        for layer in self.layers:
            result = []
            if first:
                for var, ws in zip(data, layer.neurons):
                    result.append(ws.activation(var))
                first = False
            else:
                for neuron in layer.neurons:
                    result.append(neuron.activation(data))
            data = result
        return data

    def backward(self, y, outputs):
        self.delta = y - outputs[0]
        self.error = self.delta ** 2
        print('error', self.error, 'result', outputs[0])

        index = list(range(len(self.layers)))
        index.reverse()

        first = True
        for i in index:
            if first:
                local_deltas = []
                for neuron in self.layers[i].neurons:
                    adjustments = self.for_adjustments(neuron.inputs, self.delta)
                    neuron.weights += adjustments
                    sum_weight = sum(neuron.weights)
                    for ws in neuron.weights:
                        local_delta = self.delta * ws / sum_weight
                        local_deltas.append(local_delta)
                self.delta = local_deltas
                first = False
            else:
                local_deltas = [0 for i in range(len(self.layers[i].neurons))]
                for neuron, d in zip(self.layers[i].neurons, self.delta):
                    adjustments = self.for_adjustments(neuron.weights, d)
                    neuron.weights += adjustments
                    sum_weight = sum(neuron.weights)
                    for j in range(len(neuron.weights)):
                        local_delta = neuron.weights * self.delta[j] / sum_weight
                        local_deltas[j] += local_delta


NN = NeuralNetwork([2, 2, 1])
# for i in NN.layers:
#     print('---------')
#     for j in i.neurons:
#         print(j.weights)

for i in range(500):
    k = NN.feedforward([0, 1])
    #print('data', k)
    NN.backward(0, k)
