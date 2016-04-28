
import autoencoder
import mnist_loader
import numpy as np

if __name__ == "__main__":
  # Load MNIST
  # training_input = mnist_loader.load_training_input()

  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  # Unzipping gives tuples, but we want arrays of values.
  training_input = [x for x in zip(*training_data)[0]]
  test_input = [x for x in zip(*test_data)[0]]
 
  # Make autoencoder network
  input_size = len(training_input[0])
  hidden_size = 100
  auto_enc = autoencoder.Autoencoder(input_size, hidden_size)
  auto_enc.sgd(training_input, 5, 100, 3.0, test_input)
  mse_error = auto_enc.test(test_input)
  print "mse_error: ", mse_error
