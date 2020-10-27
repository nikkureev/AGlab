import NeuralNetwork
import numpy as np

training_data = [[0, 0],
                 [0, 1],
                 [1, 0],
                 [1, 1]]
training_output = [[0], [1], [1], [0]]


NN = NeuralNetwork.NeuralNetwork(2, 2, 1)
for k in range(10000):
    # n = np.random.choice([0, 1, 2, 3])
    # NN.train(training_data[n], training_output[n])

    for i, v in zip(training_data, training_output):
        NN.train(i, v)
# input = [1, 0]
# target = [1, 1]
#output = NN.feedforward(input)
#print(output)

print('data')
print(NN.feedforward([0, 0]))
print(NN.feedforward([0, 1]))
print(NN.feedforward([1, 0]))
print(NN.feedforward([1, 1]))
