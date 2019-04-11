from neuralnetwork import NeuralNetwork
import numpy

with numpy.load('../Mnist-data-numpy-format/mnist.npz') as input_dataset:
    training_images = input_dataset['training_images']
    training_labels = input_dataset['training_labels']

layer_sizes = (784, 5, 10)

net = NeuralNetwork(layer_sizes)
predictions = net.predict(training_images)

successes = net.get_number_of_successes(predictions, training_labels)
total_predictions = len(training_labels)

print('{0} / {1}, accuracy: {2}%'.format(successes, total_predictions, (successes/total_predictions)*100))
