# Example of an Autoencoder

This is an example autoencoder implementation based on the neural network in the tutorial at http://neuralnetworksanddeeplearning.com.

## Description
A vanilla autoencoder is a feed forward neural network with a single hidden layer where the input and the desired output are the same. The intuition is that the hidden layer will learn an embedding/encoding of the data and the network will learn to encode and decode from this embedding. The hidden layer should generally not be the same size as the input/output layers because the network could simply learn the identity function. 

There are variants that further prevent the network from just learning the identity function and force it to learn more robust features, such as the denoising autoencoder that is trained to reconstruct the input from a version of the input that is corrupted by stochastic noise. For more details about this, check out this tutorial: http://deeplearning.net/tutorial/dA.html.

## Implementation Details
The MNIST dataset is used and the reconstruction error is set as the average mean squared error. Each MNIST vector corresponds to the intensities in an image of handwritten digits. This is used as both input and desired output during training.

Batch gradient descent is used to minimize the mean squared error of the output reconstruction.

To try this out, run:

`python main.py`

The output gives the epoch and batch numbers, the actual and predicted values for 10 arbitrary values from the first test example, where the 10 values are where the actual values are non-zero, and the average MSE error over the entire test set for that iteration.

Sample output:
```
epoch: 0, batch number: 0
actual -- predicted
0.33 -- 0.00
0.72 -- 1.00
0.62 -- 0.88
0.59 -- 0.00
0.23 -- 0.90
0.14 -- 0.00
0.87 -- 0.00
0.99 -- 0.06
0.99 -- 0.00
0.99 -- 0.14
0.99 -- 0.88
Average MSE error over test set: 0.300875
```

One will see that some of the predicted values approach the corresponding actual values and the MSE error should go down.

# Comparison with PCA on toy example.

Run:

'python compare_autoencoder_lsa.py'

A toy dataset of 7 points is augmented to 1000 points by jittering with Gaussian noise. This is then used to train an autoencoder with a hidden dimension of 2. PCA is also applied. These two methods are used to embed the dataset which are then plotted. 
