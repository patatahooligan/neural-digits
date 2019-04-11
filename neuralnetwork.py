import numpy

class NeuralNetwork:

    def __init__(self, layer_sizes):
        weight_shapes = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        self.weights = [numpy.random.standard_normal(s)/s[1]**0.5 for s in weight_shapes]
        self.biases = [numpy.zeros((s, 1)) for s in layer_sizes[1:]]

    def predict(self, a):
        for w, b in zip(self.weights, self.biases):
            a = self.activation(numpy.matmul(w, a) + b)
        return a

    @staticmethod
    def get_number_of_successes(predictions, labels):
        """Verify predictions using labels and return number of sucesseses"""

        prediction_success = [numpy.argmax(prediction) == numpy.argmax(label) for prediction, label in zip(predictions,labels)]
        return sum(prediction_success)

    @staticmethod
    def activation(x):
        return 1 / (1 + numpy.exp(-x))