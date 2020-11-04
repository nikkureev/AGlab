import NeuralNetwork
import NNV6
import numpy as np

training_data = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
training_output = [[0], [1], [1], [0]]


NN = NeuralNetwork.NeuralNetwork([2, 2, 1])
# for var in NN.layers_list:
#     print(var.weights.matrix)
# print('----------------------')



# NN = NNV6.NeuralNetwork(2, 2, 1)
# print(NN.weights_ih1.matrix)
# print(NN.weights_h3o.matrix)



for k in range(1000):
#     # n = np.random.choice([0, 1, 2, 3])
#     # NN.train(training_data[n], training_output[n])
#
    for i, v in zip(training_data, training_output):
        NN.train(i, v)

#
#
print('data')
print(NN.feedforward([0, 0]))
print(NN.feedforward([0, 1]))
print(NN.feedforward([1, 0]))
print(NN.feedforward([1, 1]))
