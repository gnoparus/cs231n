from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    XW1 = X.dot(W1)
    h1 = XW1 + b1
#     print('h1 = ', h1)
    f1 = np.maximum(0, h1)
#     print('f1 = ', f1)
    f1W2 = f1.dot(W2)
    h2 = f1W2 + b2
    scores = h2
#     print('X.shape = ', X.shape)
#     print('W1.shape = ', W1.shape)
#     print('XW1.shape = ', XW1.shape)
#     print('b1.shape = ', b1.shape)
#     print('h1.shape = ', h1.shape)
#     print('f1.shape = ', f1.shape)
#     print('W2.shape = ', W2.shape)
#     print('f1W2.shape = ', f1W2.shape)
#     print('b2.shape = ', b2.shape)
#     print('h2.shape = ', h2.shape)
#     print('scores.shape = ', scores.shape)
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################

    num_examples = X.shape[0]
    num_dimensions = X.shape[1]
    num_classes = W2.shape[1]

    efj = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    #   print(efj.shape)
    efy = efj[np.arange(num_examples), y]
    efy = efy.reshape(-1, 1)
    #   print(efy.shape)
    sum_efj = np.sum(efj, axis=1, keepdims=True)
    #   print(sum_efj.shape)
    loss = np.sum(-np.log(efy / sum_efj))    
    loss /= num_examples
    
    loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
  
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################

    efy_ones = np.zeros(efj.shape)    
    efy_ones[range(num_examples), y] = 1

        
    dh2 = efj / sum_efj
    dh2 -= efy_ones  
    dh2 /= num_examples

#     print('dh2.shape = ', dh2.shape)
    
    db2 = dh2.T.dot(np.ones((num_examples, 1)))[:, 0]
#     print('db2.shape = ', db2.shape)
    
    
    df1W2 = dh2
#     print('df1W2.shape = ', df1W2.shape)
    
#     print(h1)
#     print(f1)
    
    dW2 = f1.T.dot(df1W2)
#     print('dW2.shape = ', dW2.shape)
    
    df1 = df1W2.dot(W2.T)
#     print('df1.shape = ', df1.shape)
        
    dh1 = (h1 > 0) * 1 * df1
#     print('dh1.shape = ', dh1.shape)
        
    db1 = dh1.T.dot(np.ones((num_examples, 1)))[:, 0]
#     print('db1.shape = ', db1.shape)
    
    dXW1 = dh1
    
    dW1 = X.T.dot(dXW1)
   
#     print('-------------------------------------------------------------------')

    grads['b2'] = db2
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['W1'] = dW1
    
#     grads['b2'] /= num_examples
#     grads['W2'] /= num_examples
#     grads['b1'] /= num_examples
#     grads['W1'] /= num_examples
      
#     print('grads W2 = ', grads['W2'])
    
    grads['W2'] += 2 * reg * W2
    grads['W1'] += 2 * reg * W1

    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads


  def f(W, X):    
    return X.dot(W)

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
      X_batch = None
      y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      batch_idx = np.random.choice(num_train, size=batch_size, replace=True)
      X_batch = X[batch_idx, :]
      y_batch = y[batch_idx]
#       print(X_batch.shape)
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      self.params['W1'] -= learning_rate * grads['W1']
      self.params['b1'] -= learning_rate * grads['b1']
      self.params['W2'] -= learning_rate * grads['W2']
      self.params['b2'] -= learning_rate * grads['b2']
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    XW1 = X.dot(W1)
    h1 = XW1 + b1
#     print('h1 = ', h1)
    f1 = np.maximum(0, h1)
#     print('f1 = ', f1)
    f1W2 = f1.dot(W2)
    h2 = f1W2 + b2
    
    y_pred = np.argmax(h2, axis=1)
    

    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred


