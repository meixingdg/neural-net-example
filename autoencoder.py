''' 
autoencoder.py
Author: MeiXing Dong
---------------------

A module that implements an autoencoder using a feedforward neural network as a base, where the units are sigmoid units. Gradients are calculated using backpropagation.

'''

import numpy as np

class Autoencoder(object):
  '''
  sizes - list of numbers that determine the number of neurons in the various 
  layers of the network. The input and output layers should be the same size.
  ex. [12, 5, 12] is a three layer network with input and output layers of size
      12 and a single hidden layer of size 5.
  '''
  def __init__(self, sizes):
    assert(sizes[0] == sizes[-1])
    self.num_layers = len(sizes)
    self.sizes = sizes
    # One bias value for each neuron in each layer.
    self.biases = [np.random.randn(x, 1) for x in sizes[1:]]
    # One weight value connecting each neuron in one layer to each in the
    # next layer. 
    self.weights = [np.random.randn(y, x) 
		    for x, y in zip(sizes[:-1], sizes[1:])] 

  '''
  Take the input as a vector (list of numbers) and returns the output from
  passing through all of the layers.
  '''
  def feedforward(self, inputs):
    a = inputs
    for b, w in zip(self.biases, self.weights):
      # Take the input and then calculate each layer's activation.
      a = sigmoid(np.dot(w, a) + b)
    return a

  '''
  Run batch stochastic gradient descent (SGD).
  @args
  training_data - list of tuples (x, y) each representing an input and the
                  corresponding output
  epochs - number of iterations to run training for, where each iteration is
           a complete pass through all of the training data points
  batch_size - size of each batch used in SGD
  learning_coef - tunable parameter that determines how quickly the model learns
  '''
  def sgd(self, training_data, epochs, batch_size, learning_coef):
    n = len(training_data)
    for i in xrange(epochs):
      # Randomize the training data so the order that the data is given in
      # does not influence the training.
      random.shuffle(training_data)
      # Pre-process training data into batches.
      batches = [training_data[batch_index:batch_index+batch_size]
                 for batch_index in xrange(0, n, batch_size)]
      # Run an iteration of stochastic gradient descent for each batch.
      for batch in batches:
        self.sgd_helper(batch, learning_coef)
  
  '''
  Single iteration of stochastic gradient descent.
  @args
  batch - list of tuples (x, y) each representing an input and the
          corresponding output
  learning_coef - tunable parameter that determines how large each step of
                  learning is
  '''    
  def sgd_helper(self, batch, learning_coef):
    # Make zero vectors that are the same shape as the current biases and
    # weights to store values for the partial derivatives with respect
    # to each bias and weight value.
    partial_deriv_b = [np.zeros(b_.shape) for b_ in self.biases]
    partial_deriv_w = [np.zeros(w_.shape) for w_ in self.weights]
    # Go through each example in the batch and sum up the partial
    # derivatives from each with respect to b and w.
    for x, y in batch:
      single_partial_deriv_b, single_partial_deriv_w = self.backprop(x, y)
      partial_deriv_b = [pdb+spdb for pdb, spdb in zip(partial_deriv_b, single_partial_deriv_b)]
      partial_deriv_w = [pdw+spdw for pdw, spdw in zip(partial_deriv_w, single_partial_deriv_w)]
    # Update the weights and biases using the calculated partial derivatives.
    self.biases = [b-(learning_coef/len(batch))*deriv_b
                   for b, deriv_b in zip(self.biases, partial_deriv_b)]
    self.weights = [w-(learning_coef/len(batch))*deriv_w
                    for w, deriv_w in zip(self.weights, partial_deriv_w)]

  '''
  Return a tuple (partial_deriv_b, partial_deriv_w) representing the gradient
  for the cost function. The partial derivatives are of the cost function with
  respect to each bias and weight.
  '''
  def backprop(self, x, y):
    partial_deriv_b = [np.zeros(b.shape) for b in self.biases]
    partial_deriv_w = [np.zeros(w.shape) for w in self.biases]

    # Feed input forward to calculate activations and z = wx + b of each layer.
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)      

    # TODO: finish





