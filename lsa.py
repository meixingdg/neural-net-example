# Author: MeiXing Dong

import numpy as np
import mnist_loader
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def visualize():
  training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
  # Unzipping gives tuples, but we want arrays of values.
  training_input = [x.transpose()[0] for x in zip(*training_data)[0]]
  test_input = [x.transpose()[0] for x in zip(*test_data)[0]]
  # Get the y values.
  test_target = [y for y in zip(*test_data)[1]]

  # Apply SVD to the training input.
  u, s, v = np.linalg.svd(training_input, full_matrices=False)
  print u.shape
  print s.shape
  print v.shape
  
  print "Generating embeddings..."
  #print v[0]
  print v[0].shape
  embeddings = [np.dot(test_inp, np.transpose(v[:10][:])) for test_inp in test_input]
  print embeddings[0].shape
  
  # Do dimensionality reduction into 2 dimensions.
  print "Performing dimensionality reduction using t-sne..."
  tsne = TSNE()
  reduced_vecs = tsne.fit_transform(embeddings)
  print reduced_vecs[0]

  # Graph all of the points, where points corresponding to the same digit will have the same color.
  colors = ['r', 'b', 'g', 'c', 'm', 'k', 'y', (.2, .2, .2), (.4, 0, .5), (.8, .2, 0)]
  red_patch = mpatches.Patch(color='red', label='1')
  patches = [mpatches.Patch(color=colors[i], label='%i'% i) for i in range(len(colors))]
  plt.legend(handles=patches)
  for i in range(len(reduced_vecs)):
    plt.plot([reduced_vecs[i][0]], [reduced_vecs[i][1]], 'o', color=colors[test_target[i]])
  plt.show()


if __name__ == "__main__":
  visualize()
