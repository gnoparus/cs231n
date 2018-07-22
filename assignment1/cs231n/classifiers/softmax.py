import numpy as np
from random import shuffle

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
    
  num_examples = X.shape[0]
  num_dimensions = X.shape[1]
  num_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
#   for i in range(num_examples):        
        
#         efy = np.exp(f(W, X[i])[y[i]])
#         efj = np.exp(f(W, X[i]))
#         sum_efj = np.sum(efj)
#         dW[:, y[i]] += efj / sum_efj
#         loss += -np.log(efy / sum_efj)

  ef = np.exp(f(W, X))
#   print(ef.shape)
  efy = ef[np.arange(num_examples), y]
  efy = efy.reshape(-1, 1)
#   print(efy.shape)
  efj = np.exp(f(W, X))
#   print(efj.shape)
  sum_efj = np.sum(efj, axis=1, keepdims=True)
#   print(sum_efj.shape)
  loss = np.sum(-np.log(efy / sum_efj))
  
    
#   dW[:, y] = efj / sum_efj
    
#   print(dW[0:5])
  efy_ones = np.zeros(ef.shape)
  efy_ones[range(num_examples), y] = 1
    
  dW = X.T.dot(ef / sum_efj)
  
  dW -= X.T.dot(efy_ones)
  
  loss /= num_examples
  dW /= num_examples
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  
  num_examples = X.shape[0]
  num_dimensions = X.shape[1]
  num_classes = W.shape[1]

#   for i in range(num_examples):        
        
#         efy = np.exp(f(W, X[i])[y[i]])
#         efj = np.exp(f(W, X[i]))
#         sum_efj = np.sum(efj)
#         dW[:, y[i]] += efj / sum_efj
#         loss += -np.log(efy / sum_efj)

  ef = np.exp(f(W, X))
#   print(ef.shape)
  efy = ef[np.arange(num_examples), y]
  efy = efy.reshape(-1, 1)
#   print(efy.shape)
  efj = np.exp(f(W, X))
#   print(efj.shape)
  sum_efj = np.sum(efj, axis=1, keepdims=True)
#   print(sum_efj.shape)
  loss = np.sum(-np.log(efy / sum_efj))
  
    
#   dW[:, y] = efj / sum_efj
    
#   print(dW[0:5])
  efy_ones = np.zeros(ef.shape)
  efy_ones[range(num_examples), y] = 1
    
  dW = X.T.dot(ef / sum_efj)
  
  dW -= X.T.dot(efy_ones)
  
  loss /= num_examples
  dW /= num_examples
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def f(W, X):
    
    return X.dot(W)
