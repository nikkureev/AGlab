import MathSource as MS


def layer_activation(layer, inputs, bias):
    output = MS.Matrix.multiply(layer, inputs)
    output.add(bias)
    output.map(MS.sigmoid)
    return output

def gradient_calculation(inputs, errors, learning_rate):
    gradients = inputs.map(MS.sigmoid_deriv)
    gradients = MS.Matrix.multiply(gradients, errors)
    gradients = MS.Matrix.multiply(gradients, learning_rate)
    return gradients

def applying_adjustments(inputs, gradients, layer, bias):
    hidden_T = MS.Matrix.transpose(inputs)
    weights_ho_deltas = MS.Matrix.multiply(gradients, hidden_T)
    layer.add(weights_ho_deltas)
    bias.add(gradients)


class NeuralNetwork():

    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.weights_ih = MS.Matrix(self.hidden_nodes, self.input_nodes)
        self.weights_ho = MS.Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomize()
        self.weights_ho.randomize()

        self.bias_h = MS.Matrix(self.hidden_nodes, 1)
        self.bias_o = MS.Matrix(self.output_nodes, 1)
        self.bias_h.randomize()
        self.bias_o.randomize()

        self.learning_rate = 0.1


    def feedforward(self, input_array):

        inputs = MS.fromArray(input_array)

        hidden = layer_activation(self.weights_ih, inputs, self.bias_h)
        outputs = layer_activation(self.weights_ho, hidden, self.bias_o)

        return outputs.toArray()


    def train(self, input_array, target_array):

        inputs = MS.fromArray(input_array)
       
        hidden = layer_activation(self.weights_ih, inputs, self.bias_h)
        outputs = layer_activation(self.weights_ho, hidden, self.bias_o)

        targets = MS.fromArray(target_array)
        output_errors = MS.subtract(targets, outputs)

        gradients = gradient_calculation(outputs, output_errors, self.learning_rate)
        applying_adjustments(hidden, gradients, self.weights_ho, self.bias_o)

        weights_ho_t = MS.Matrix.transpose(self.weights_ho)
        hidden_errors = MS.Matrix.multiply(weights_ho_t, output_errors)

        hidden_gradient = gradient_calculation(hidden, hidden_errors, self.learning_rate)
        applying_adjustments(inputs, hidden_gradient, self.weights_ih, self.bias_h)
