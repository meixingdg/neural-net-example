''' 
autoencoder.py
Author: MeiXing Dong
---------------------

A module that implements an autoencoder using a feedforward neural network as a base, where the units are sigmoid units. Gradients are calculated using backpropagation.

'''

import numpy as np
import random

class Autoencoder(object):
  '''
  sizes - list of numbers that determine the number of neurons in the various 
  layers of the network. The input and output layers should be the same size.
  ex. [12, 5, 12] is a three layer network with input and output layers of size
      12 and a single hidden layer of size 5.
  '''
  def __init__(self, input_size, hidden_size):
    self.num_layers = 3
    self.sizes = [input_size, hidden_size, input_size]
    # One bias value for each neuron in each layer.
    self.biases = [np.random.randn(x, 1) for x in self.sizes[1:]]
    # One weight value connecting each neuron in one layer to each in the
    # next layer. 
    self.weights = [np.random.randn(y, x) 
		    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

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
  training_data - list of inputs x (which is also the desired output)
  epochs - number of iterations to run training for, where each iteration is
           a complete pass through all of the training data points
  batch_size - size of each batch used in SGD
  learning_coef - tunable parameter that determines how quickly the model learns
  '''
  def sgd(self, training_data, epochs, batch_size, learning_coef, test_data=None):
    track_progress = False
    if test_data:
      track_progress = True
    n = len(training_data)
    for i in xrange(epochs):
      print(i)
      # Randomize the training data so the order that the data is given in
      # does not influence the training.
      random.shuffle(training_data)
      # Pre-process training data into batches.
      batches = [training_data[batch_index:batch_index+batch_size]
                 for batch_index in xrange(0, n, batch_size)]
      # Run an iteration of stochastic gradient descent for each batch.
      j = 0
      for batch in batches:
        print(j)
        j = j + 1
        self.sgd_helper(batch, learning_coef)
        print "Average MSE error over test set: %f" % self.test(test_data)
  
  '''
  Single iteration of stochastic gradient descent.
  @args
  batch - list of x each representing an input (is also the desired output)
  learning_coef - tunable parameter that determines how large each step of
                  learning is
  '''    
  def sgd_helper(self, batch, learning_coef):
    # Make zero vectors that are the same shape as the current biases and
    # weights to store values for the partial derivatives with respect
    # to each bias and weight value.
    partial_deriv_b = np.array([np.zeros(b_.shape) for b_ in self.biases])
    partial_deriv_w = np.array([np.zeros(w_.shape) for w_ in self.weights])
    # Go through each example in the batch and sum up the partial
    # derivatives from each with respect to b and w.
    for x in batch: 
      single_partial_deriv_b, single_partial_deriv_w = self.backprop(x, x)
      assert(np.array(single_partial_deriv_w).shape == np.array(partial_deriv_w).shape)
      partial_deriv_b = partial_deriv_b + np.array(single_partial_deriv_b)
      partial_deriv_w = partial_deriv_w + single_partial_deriv_w
    # Update the weights and biases using the calculated partial derivatives.
    self.biases = np.array(self.biases) - (learning_coef/len(batch))*partial_deriv_b
    self.weights = np.array(self.weights) - (learning_coef/len(batch))*partial_deriv_w

  '''
  Return a tuple (partial_deriv_b, partial_deriv_w) representing the gradient
  for the cost function. The partial derivatives are of the cost function with
  respect to each bias and weight.
  Cost function is the MSE: C = 1/2n*\sum_n||y - output||^2
  @args
  x - single input vector
  y - single corresponding output vector
  '''
  def backprop(self, x, y):
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    partial_deriv_b = [np.zeros(b.shape) for b in self.biases]
    partial_deriv_w = [np.zeros(w.shape) for w in self.weights]

    # Feed input forward to calculate activations and z = wx + b of each layer.
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation) 

    # Go through each layer backwards and propagate the error.
    L = self.num_layers-1
    errors = (activations[L] - y)*sigmoid_deriv(zs[L-1])
    # There's one less set of weights/biases than the number of layers
    # because the input layer doesn't have weights or biases. 
    partial_deriv_b[L-1] = errors
    partial_deriv_w[L-1] = np.dot(errors, np.transpose(activations[L-1]))
    for l in xrange(L-2, 0, -1):
      errors = np.dot(np.transpose(self.weights[l+1]), errors) \
               * sigmoid_deriv(zs[l])
      partial_deriv_b[l] = errors
      partial_deriv_w[l] = np.dot(errors, np.transpose(activations[l]))
    
    return (partial_deriv_b, partial_deriv_w)
  
  ''' Return the squared error.
  '''
  def mse(self, x, y):
    assert(x.shape == y.shape)
    return sum((x - y)*(x - y))/len(x)

  '''
  Feed every test input forward through the network and calculate the error
  of the output. Each wrongly classified digit is an error.
  '''
  def test(self, test_data):
    output = [self.feedforward(x) for x in test_data]
    error = sum([self.mse(x, y) for (x, y) in zip(test_data, output)])
    return error/len(test_data)

def sigmoid(x):
  return 1.0/(1.0 + np.exp(-x))

def sigmoid_deriv(x):
  return sigmoid(x)*(1-sigmoid(x))
