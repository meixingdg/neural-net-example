# Author: MeiXing Dong

import autoencoder
import numpy as np
import matplotlib.pyplot as plt
import pickle

def main():
  # Original data, where each row is a document and each column represents
  # a characteristic of the documents. 
  data = [[1, 1, 1, 0, 0],
        [2, 2, 2, 0, 0],
        [1, 1, 1, 0, 0],
        [5, 5, 5, 0, 0],
        [0, 0, 0, 2, 2],
        [0, 0, 0, 3, 3],
        [0, 0, 0, 1, 1]]
  
  # Augment dataset to 1000 elements by adding Gaussian noise.
  data_aug = []
  i = 0
  while i < 1000:
    for x in data:
      data_aug.append([num+np.random.normal() for num in x])
      i += 1
  data_aug = np.array(data_aug)

  # Do PCA to get document matrix for embedding the values.
  u, s, v = np.linalg.svd(data_aug, full_matrices=False)
  # Output U, S, and V
  print "Shape of U: ", u.shape
  print "Shape of S: ", s.shape
  print "Shape of V: ", v.shape
  v = v[:2][:]
  print "Shape of truncated V: ", v.shape

  # Modify format of data to match expected input of autoencoder.
  data_aug_modified = []
  for point in data_aug:
    data_aug_modified.append([[num] for num in point])
  data_aug_modified = np.array(data_aug_modified)

  # TODO: load trained autoencoder if already exists
  # Train autoencoder with hidden dimension 2 if a trained autoencoder doesn't already exist.
  print "Training autoencoder..."
  auto_enc = autoencoder.Autoencoder(5, 2)
  auto_enc.sgd(data_aug_modified, 100, 100, 3.0, data_aug_modified)
  
  # Save trained autoencoder to file.
  print "Saving trained autoencoder..."
  outfile = open("trained_autoencoder_toy.pkl", "w")
  pickle.dump(auto_enc, outfile)
  mse_error = auto_enc.test(data_aug_modified)
  print "mse_error: ", mse_error

  # Output learned hidden weights.
  print "Autoencoder hidden weights: ", auto_enc.weights[1]
  print "PCA truncated U: ", u[:][:2]
  print "PCA truncated S: ", s[:2][:2]
  print "PCA truncated V: ", np.transpose(v[:2][:])
  
  fig = plt.figure()
  plt.subplot(211)
  plt.title("PCA")
  # TODO: see if values between the two match?
  # Plot embeddings of the documents from the autoencoder and PCA on same graph.
  # Embed data using PCA.
  print "Embedding data using PCA..."
  pca_embed = np.dot(data_aug, v.transpose())
  print pca_embed[0]
  plt.plot([p[0] for p in pca_embed], [p[1] for p in pca_embed], 'bx')
  #plt.show()

  # Embed data using the autoencoder.
  print "Embedding data using autoencoder..."
  autoenc_embed = [auto_enc.feedforward(p, embed=True).transpose()[0] for p in data_aug_modified]
  plt.subplot(212)
  plt.title("Autoencoder") 
  for i in range(len(autoenc_embed)):
    plt.plot([autoenc_embed[i][0]], [autoenc_embed[i][1]], 'ro')
  
  plt.show()

  return

if __name__ == "__main__":
  main()
