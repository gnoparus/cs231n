import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]  
  num_train = X.shape[0]
  loss = 0.0
  

  for i in range(num_train):
    count_incorrect = 0
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      margin = scores[j] - correct_class_score + 1 # note delta = 1

      if j == y[i]:        
        continue
      if margin > 0:
        loss += margin        
        dW[:, j] += X[i].T
        count_incorrect += 1

    dW[:, y[i]] += - count_incorrect * X[i].T
    
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  dWyi = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_dimensions = W.shape[0]
  scores = X.dot(W)
#   print(y.shape)
#   print(scores.shape)
#   print(scores[0:5])
#   print(y[0:5])
  correct_class_scores = scores[np.arange(0, num_train), y].reshape(-1, 1)
#   print(correct_class_scores.shape)
#   print(correct_class_scores[0:5])
   
    
  margins = np.maximum(0, scores - correct_class_scores + 1)
#   print(margins.shape)
#   print(margins[0:5])
  

  
#   dWyi = np.ones(margins.shape) * margins_idx
#   print(dWyi.shape)
#   print(dWyi[0:5])
#   print(margins_idx.shape)
#   print(margins[margins_idx].shape)

  margins[np.arange(0, num_train), y] = 0
#   print(margins[0:5])
  
  margins_ones = np.ones(margins.shape) * (margins > 0)
#   print(margins_ones.shape)
#   print(margins_ones[0:5])

  correct_class_ones = np.zeros(margins.shape)
  correct_class_ones[np.arange(num_train), y] = 1
#   print(correct_class_ones.shape)
#   print(correct_class_ones[0:5])    

  loss = np.sum(margins)

#   print((margins > 0)[0:5])
#   print(X.T[0:5])
#   print(X.T.shape)
  
  dWj = X.T.dot(margins_ones)
#   print(dWj[0:5])
  margins_count = np.count_nonzero(margins, axis=1)
  
#   print(margins_count.shape)
  dWyi = -X.T.dot(margins_count.reshape(-1, 1) * correct_class_ones)
#   print(dW[0:5])


#   dW[np.arange(0, num_dimensions), y] += - np.count(margins > 0) * X.T
  dW = dWyi + dWj

  loss /= num_train
  dW /= num_train
    
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
