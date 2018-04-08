import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  def softmax_prob(normalized_logits, normalized_logits_sum, class_number):
      numerator = np.exp(normalized_logits[class_number])
      activation = numerator / normalized_logits_sum
      return activation

  y_pred = X.dot(W)
  for i in range(num_train):
      normalized_current_sample = y_pred[i] - np.max(y_pred[i])
      normalized_current_sample_sum = np.sum(np.exp(normalized_current_sample))
      softmax_prob_current_sample = softmax_prob(normalized_current_sample,
                                                 normalized_current_sample_sum,
                                                 y[i])
      loss += -np.log(softmax_prob_current_sample) # cross entropy
      for j in range(num_classes):
          softmax_prob_current_j = softmax_prob(normalized_current_sample,
                                                normalized_current_sample_sum,
                                                j)
          dW[:, j] += (softmax_prob_current_j - (j == y[i])) * X[i]
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW /= num_train
  dW += reg*W
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = X.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  y_pred = X.dot(W)
  y_pred = y_pred - np.max(y_pred, axis=1).reshape(-1, 1)
  y_pred = np.exp(y_pred[np.arange(num_train), y]) / np.sum(np.exp(y_pred), axis=1)
  loss = -np.log(y_pred)
  loss = np.sum(loss)
  loss /= num_train

  dW = X.dot(W)
  dW = np.exp(dW) / np.sum(np.exp(dW), axis=1).reshape(-1, 1)
  dW[np.arange(num_train), y] -= 1
  dW = dW * X.T[:, :, None]
  dW = np.sum(dW, axis=1)
  dW /= num_train
  dW += reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
