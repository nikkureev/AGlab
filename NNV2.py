import MathSource as MS


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

        self.learning_rate = 1


    def feedforward(self, input_array):

        inputs = MS.fromArray(input_array)
        hidden = MS.Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)
        hidden.map(MS.sigmoid)

        output = MS.Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(MS.sigmoid)

        return output.toArray()


    def train(self, input_array, target_array):

        # Generating the Hidden Outputs
        inputs = MS.fromArray(input_array)
        hidden = MS.Matrix.multiply(self.weights_ih, inputs)
        hidden.add(self.bias_h)

        # Activation function
        hidden.map(MS.sigmoid)

        # Generating the output's output
        outputs = MS.Matrix.multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(MS.sigmoid)

        # Backward, counting errors
        targets = MS.fromArray(target_array)
        output_errors = MS.subtract(targets, outputs)

        # Generating adjustments
        gradients = outputs.map(MS.sigmoid_deriv)
        gradients = MS.Matrix.multiply(gradients, output_errors)
        gradients = MS.Matrix. multiply(gradients, self.learning_rate)


        # Applying adjustments
        hidden_T = MS.Matrix.transpose(hidden)
        weights_ho_deltas = MS.Matrix.multiply(gradients, hidden_T)
        self.weights_ho.add(weights_ho_deltas)
        self.bias_o.add(gradients)

        weights_ho_t = MS.Matrix.transpose(self.weights_ho)
        hidden_errors = MS.Matrix.multiply(weights_ho_t, output_errors)

        hidden_gradient = hidden.map(MS.sigmoid_deriv)
        hidden_gradient = MS.Matrix.multiply(hidden_gradient, hidden_errors)
        hidden_gradient = MS.Matrix.multiply(hidden_gradient, self.learning_rate)


        inputs_T = MS.Matrix.transpose(inputs)
        weights_ih_deltas = MS.Matrix.multiply(hidden_gradient, inputs_T)
        self.weights_ih.add(weights_ih_deltas)
        self.bias_h.add(hidden_gradient)
