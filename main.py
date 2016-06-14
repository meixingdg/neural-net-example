# Author: MeiXing Dong

import autoencoder
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import mnist_loader
import numpy as np
import pickle
import sklearn.manifold

def main():
  # Load MNIST
  # training_input = mnist_loader.load_training_input()

  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  # Unzipping gives tuples, but we want arrays of values.
  training_input = [x for x in zip(*training_data)[0]]
  test_input = [x for x in zip(*test_data)[0]]
  print type(training_input[0])
  #print(training_input[0][:10])
  #print(autoencoder.add_noise(training_input[0])[:10])

  # Make autoencoder network
  input_size = len(training_input[0])
  hidden_size = 100
  auto_enc = autoencoder.Autoencoder(input_size, hidden_size)
  #auto_enc.sgd(training_input, 5, 100, 3.0)
  auto_enc.sgd(training_input, 5, 100, 3.0, test_input)
  # Save the trained autoencoder to file.
  outfile = open("trained_autoencoder.pkl", "w")
  pickle.dump(auto_enc, outfile)
  mse_error = auto_enc.test(test_input)
  print "mse_error: ", mse_error

def visualize():
  infile = open("trained_autoencoder.pkl")
  auto_enc = pickle.load(infile)

  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  # Unzipping gives tuples, but we want arrays of values.
  training_input = [x for x in zip(*training_data)[0]]
  test_input = [x for x in zip(*test_data)[0]]
  # Get the y values.
  test_target = [y for y in zip(*test_data)[1]]
  print test_target[0]

  # Encode all of the MNIST test set using the autoencoder.
  # TODO: get rid of debugging, do all points not just 50
  print "Encoding MNIST using autoencoder..."
  autoencoder_encoded_vecs = [auto_enc.feedforward(test_inp, embed=True).transpose()[0] for test_inp in test_input]
  print len(autoencoder_encoded_vecs)
  # print autoencoder_encoded_vecs[0]
  # print autoencoder_encoded_vecs[0].shape
  # Do dimensionality reduction into 2 dimensions using t-sne.
  print "Performing dimensionality reduction using t-sne..."
  tsne = sklearn.manifold.TSNE()
  reduced_vecs = tsne.fit_transform(autoencoder_encoded_vecs)    
  print reduced_vecs[0]
  #plt.plot([p[0] for p in reduced_vecs[:30]], [p[1] for p in reduced_vecs[:30]], 'ro')

  # Graph all of the points, where points corresponding to the same digit will have the same color.
  colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y', (.2, .2, .2), (.4, 0, .5), (.8, .2, 0)]
  red_patch = mpatches.Patch(color='red', label='1')
  patches = [mpatches.Patch(color=colors[i], label='%i'% i) for i in range(len(colors))]
  plt.legend(handles=patches)
  for i in range(len(reduced_vecs)):
    plt.plot([reduced_vecs[i][0]], [reduced_vecs[i][1]], 'o', color=colors[test_target[i]]) 
  plt.show()
  
if __name__ == "__main__":
  main()
  visualize()
  
