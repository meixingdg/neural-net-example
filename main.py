
import autoencoder
import mnist_loader
import numpy as np

if __name__ == "__main__":
  # Load MNIST
  training_input = mnist_loader.load_training_input()
  
  # Make autoencoder network
  input_size = len(training_input[0])
  hidden_size = 100
  auto_enc = autoencoder.Autoencoder(input_size, hidden_size)
  auto_enc.sgd(training_input, 5, 100, 3.0)
  
