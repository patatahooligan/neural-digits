from neuralnetwork import NeuralNetwork
import numpy

layer_sizes = (3, 5, 10)
input = numpy.ones((layer_sizes[0], 1))

net = NeuralNetwork(layer_sizes)
prediction = net.predict(input)

print(prediction)