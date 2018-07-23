from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_nd = np.reshape(x, (x.shape[0], -1))
    out = x_nd.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    
    dx = dout.dot(w.T)
    dx = dx.reshape(x.shape)

    dw = np.reshape(x, (x.shape[0], -1)).T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x.copy()
    out[x <= 0] = 0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    
    dx = (x > 0) * 1 * dout
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        # 
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        sample_sd = np.sqrt(sample_var + eps)
        cache = {}
        
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        xhat = x.copy().astype(np.float64)
        xhat -= sample_mean
        xhat /= (np.sqrt(sample_var + eps))
#         print('normalized out = ', out)
        out = xhat.copy()
    
        out *= gamma
        out += beta
        
        cache['gamma'] = gamma.copy()
        cache['xhat'] = xhat.copy()
        cache['sample_mean'] = sample_mean.copy()
        cache['sample_var'] = sample_var.copy()
        cache['sample_sd'] = sample_sd.copy()
        cache['x'] = x.copy()
        cache['eps'] = eps
        cache['N'] = N
        
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        
        
        out = x.copy()
        out -= running_mean
        out /= (np.sqrt(running_var) + eps)
        
        out *= gamma
        out += beta
                
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
            
    x = cache['x'].astype(np.float64)
    N = cache['N']
    sample_mean = cache['sample_mean'].astype(np.float64)
    sample_var = cache['sample_var'].astype(np.float64)
    sample_sd = cache['sample_sd'].astype(np.float64)
    eps = cache['eps']
    gamma = cache['gamma'].astype(np.float64)
    xhat = cache['xhat'].astype(np.float64)
    
    dbeta = dout.sum(axis=0)            
    dgamma = np.sum(dout * xhat, axis=0)       

    dxhat = dout * gamma 
    
    dsample_var = np.sum((dxhat * (x - sample_mean) * -0.5 * (np.power(((sample_sd ** 2.0) + eps), (-3.0/2.0)))), axis=0)
#     print('sample_var   = ', sample_var)
#     print('dsample_var  = ', dsample_var)
#     print(type(dsample_var[0]))

    
    dsample_mean = np.sum(dxhat * -1.0 / np.sqrt((sample_sd ** 2.0) + eps), axis=0) + dsample_var * np.sum(-2.0 * (x - sample_mean), axis=0) / N
#     print('sample_mean  = ', sample_mean)
#     print('dsample_mean = ', dsample_mean)
        
    dx = (dxhat / np.sqrt((sample_sd ** 2.0) + eps)) + (dsample_var * 2.0 * (x - sample_mean) / N) + (dsample_mean / N)
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
     
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    
    x = cache['x']
    N = cache['N']
    sample_mean = cache['sample_mean']
    sample_var = cache['sample_var']
    sample_sd = cache['sample_sd']
    eps = cache['eps']
    gamma = cache['gamma']
    xhat = cache['xhat']
    
    dbeta = dout.sum(axis=0)            
    dgamma = np.sum(dout * xhat, axis=0)       
    
    df = dout * gamma 
    
    dx = (1. / N) * gamma * (sample_var + eps) ** (-1. / 2.) * (N * dout - np.sum(dout, axis=0) - (x - sample_mean) * (sample_var + eps) ** (-1.0) * np.sum(dout * (x - sample_mean), axis=0))
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.
    
    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    
#     print('x.shape   = ', x.shape)
    mu = x.mean(axis=1).reshape(1, -1).T
    sigma = x.var(axis=1).reshape(1, -1).T
    sd = np.sqrt(sigma + eps)
    
    xhat = (x - mu) / sd    
#     print('xhat.shape  = ', xhat.shape)
#     print('gamma.shape = ', gamma.shape)
#     print('beta.shape  = ', beta.shape)
    
    gamma = gamma.reshape(1, -1).T
    beta = beta.reshape(1, -1).T
    
    out = (xhat * gamma) + beta    
#     print('out.shape  = ', out.shape)
    
    cache = eps, mu, sigma, sd, xhat, x, gamma, beta    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
        
    eps, mu, sigma, sd, xhat, x, gamma, beta = cache
        
    D = x.shape[1]
    dbeta = dout.sum(axis=0)            
    dgamma = np.sum(dout * xhat, axis=0)

    dxhat = dout * gamma     
   
    dsigma = np.sum((dxhat * (x - mu) * -0.5 * (np.power(((sd ** 2.0) + eps), (-3.0/2.0)))), axis=1, keepdims=True)
#     print('sample_var   = ', sample_var)
#     print('dsample_var  = ', dsample_var)
#     print(type(dsample_var[0]))

    
    dmu = np.sum(dxhat * -1.0 / np.sqrt((sd ** 2.0) + eps), axis=1, keepdims=True) + dsigma * np.sum(-2.0 * (x - mu), axis=1, keepdims=True) / D
#     print('sample_mean  = ', sample_mean)
#     print('dsample_mean = ', dsample_mean)
        
    dx = (dxhat / np.sqrt((sd ** 2.0) + eps)) + (dsigma * 2.0 * (x - mu) / D) + (dmu / D)
    
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = np.random.rand(*x.shape) * 1
        mask = mask > (1 - p)
        out = x * mask
        out = out / p
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        dx /= dropout_param['p']
        
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
#     print('WW = ', WW)
#     print('HH = ', WW)
    
    Hprime = 1 + (H + 2 * pad - HH) / stride
    Wprime = 1 + (W + 2 * pad - WW) / stride
    
    out = np.zeros((N, F, int(Hprime), int(Wprime)))
#     print('out.shape = ', out.shape)
    
#     print('x.shape    = ', x.shape)
#     print('x          = ', x)
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0))
#     print('x_pad.shape = ', x_pad.shape)
#     print('x_pad       = ', x_pad)

    for n in np.arange(N):
        for f in np.arange(F):
            for h in np.arange(int(Hprime)):
                for wi in np.arange(int(Wprime)):
#                     print(n, f, h, w)
#                     out[n, f, h, w] = 0
                    projected_h = h * stride
#                     print('projected_h  = ', projected_h)
                    projected_wi = wi * stride
#                     print('projected_wi = ', projected_wi)
                    w_filter = w[f, :, :, :]
#                     print('w_filter.shape = ', w_filter.shape)
                    projected_x = x_pad[n, :, projected_h:projected_h+HH, projected_wi:projected_wi+WW]
#                     print('projected_x.shape = ', projected_x.shape)
                    out[n, f, h, wi] = np.sum(w_filter * projected_x) + b[f] 

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

#     print('dout.shape = ', dout.shape)
#     print('dx.shape   = ', dx.shape)
#     print('dw.shape   = ', dw.shape)
#     print('db.shape   = ', db.shape)
    
    db = dout.sum(axis=3)                
    db = db.sum(axis=2)            
    db = db.sum(axis=0)            
    print('db.shape   = ', db.shape)
    
    
    
#     dw = np.sum(dout * x, axis=0)

    # dout.shape
    Hprime = 1 + (H + 2 * pad - HH) / stride
    Wprime = 1 + (W + 2 * pad - WW) / stride
    
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=(0))
    dx_pad = np.zeros_like(x_pad)
    
    for n in np.arange(N):
        for f in np.arange(F):
            for h in np.arange(int(Hprime)):
                for wi in np.arange(int(Wprime)):
#                     print(n, f, h, w)
                    projected_h = h * stride
#                     print('projected_h  = ', projected_h)
                    projected_wi = wi * stride
#                     print('projected_wi = ', projected_wi)
                    projected_x = x_pad[n, :, projected_h:projected_h+HH, projected_wi:projected_wi+WW]
                    projected_dx = dx_pad[n, :, projected_h:projected_h+HH, projected_wi:projected_wi+WW]
#                     print('projected_x.shape = ', projected_x.shape)
                    dw[f, :, :, :] += dout[n, f, h, wi] * projected_x
                    projected_dx += dout[n, f, h, wi] * w[f, :, :, :]
    
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N, C, H, W = x.shape
            
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']

    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)
    
    out = np.zeros((N, C, Hprime, Wprime))
    posh = np.zeros((N, C, Hprime, Wprime)).astype(int)
    posw = np.zeros((N, C, Hprime, Wprime)).astype(int)

    for n in np.arange(N):
        for h in np.arange(int(Hprime)):
            for wi in np.arange(int(Wprime)):            

                projected_h = h * stride
                projected_wi = wi * stride
                projected_x = x[n, :, projected_h:projected_h+pool_height, projected_wi:projected_wi+pool_width]
                
                out[n, :, h, wi] = projected_x.max(axis=-1).max(axis=-1)
                
                posh[n, :, h, wi] = projected_x.max(axis=-1).argmax(axis=-1)
                posw[n, :, h, wi] = projected_x.max(axis=-2).argmax(axis=-1)
                
    
#     print(    x  [0, 0, :, :])
#     print(    out[0, 0, :, :])
#     print(   posh[0, 0, :, :])
#     print(   posw[0, 0, :, :])
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, posh, posw)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################

    x, pool_param, posh, posw = cache
    N, C, H, W = x.shape
            
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']

    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)
    
    dx = np.zeros((N, C, H, W))

    for n in np.arange(N):
        for h in np.arange(int(Hprime)):
            for wi in np.arange(int(Wprime)):            

                projected_h = h * stride
                projected_wi = wi * stride
#                 projected_x = x[n, :, projected_h:projected_h+pool_height, projected_wi:projected_wi+pool_width]
#                 projected_dx = dx[n, :, projected_h:projected_h+pool_height, projected_wi:projected_wi+pool_width]
                
#                 projected_x = x[n, :, projected_h:projected_h+pool_height, projected_wi:projected_wi+pool_width]
                
#                 print(posh[n, :, h, wi])
#                 print(posw[n, :, h, wi])
        
#                 projected_dx[:, posh[n, :, h, wi], posw[n, :, h, wi]] = dout[n, :, h, wi]
#                 dx[n, :, projected_h+posh[n, :, h, wi], projected_wi+posw[n, :, h, wi]] = dout[n, :, h, wi]
                dx[n, np.arange(C), projected_h+posh[n, np.arange(C), h, wi], projected_wi+posw[n, np.arange(C), h, wi]] = dout[n, np.arange(C), h, wi]
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    N, C, H, W = x.shape
    cache = {}
    
    out = np.zeros((N, C, H, W))
    
#     print('x.shape    = ', x.shape)
    for c in np.arange(C):
    
        xc_re = x[:, c, :, :].reshape(x.shape[0], -1)
#         print('xc_re.shape = ', xc_re.shape)
        out_re, cache[c] = batchnorm_forward(xc_re, gamma[c], beta[c], bn_param)
        out[:, c, :, :] = out_re.reshape(x.shape[0], x.shape[2], x.shape[3])
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    
    N, C, H, W = dout.shape
    
    dx = np.zeros((N, C, H, W))
    dgamma = np.zeros(C)
    dbeta = np.zeros(C)
    
#     print('x.shape    = ', x.shape)
    for c in np.arange(C):
    
        doutc_re = dout[:, c, :, :].reshape(dout.shape[0], -1)
#         print('doutc_re.shape = ', doutc_re.shape)
        dx_re, dgamma_re, dbeta_re = batchnorm_backward_alt(doutc_re, cache[c])
        
        dgamma[c] = np.sum(dgamma_re)
        dbeta[c] = np.sum(dbeta_re)
        dx[:, c, :, :] += dx_re.reshape(dout.shape[0], dout.shape[2], dout.shape[3])    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry 
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                # 
    ###########################################################################

    N, C, H, W = x.shape
    cache = {}
    ln_cache = {}
    
    out = np.zeros((N, C, H, W))
    
    cg = C // G
#     print('cg = ', cg)
    
#     print('x.shape    = ', x.shape)
    for n in np.arange(N):
        for g in np.arange(G):

            projected_g = cg*g
            xc_re = x[n, projected_g:projected_g+cg, :, :].reshape(cg, -1)
#             print('xc_re.shape  = ', xc_re.shape)

    #         out_re, cache[c] = batchnorm_forward(xc_re, gamma[c], beta[c], bn_param)

            # out, cache = layernorm_forward(x, gamma, beta, ln_param)
            out_re, ln_cache[n, g] = layernorm_forward(xc_re, gamma[0, projected_g:projected_g+cg, 0, 0], beta[0, projected_g:projected_g+cg, 0, 0], gn_param)

#             print('out_re.shape = ', out_re.shape)

            out[n, projected_g:projected_g+cg, :, :] = out_re.reshape(cg, x.shape[2], x.shape[3])
    
        cache = ln_cache, G
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################

    
    N, C, H, W = dout.shape
    
    dx = np.zeros((N, C, H, W))
    dgamma = np.zeros((1,C,1,1))
    dbeta = np.zeros((1,C,1,1))
    ln_cache, G = cache 
    xhat = np.zeros((N, C, H, W))
    
    cg = C // G
#     print('cg = ', cg)

    
#     print('x.shape    = ', x.shape)
    for n in np.arange(N):
        for g in np.arange(G):

            projected_g = cg*g
            doutcg_re = dout[n, projected_g:projected_g+cg, :, :].reshape(cg, -1)
#             print('xc_re.shape  = ', xc_re.shape)

#             out_re, cache = layernorm_forward(doutc_re, gamma[0, projected_g:projected_g+cg, 0, 0], beta[0, projected_g:projected_g+cg, 0, 0], gn_param)

            dx_re, dgamma_re, dbeta_re = layernorm_backward(doutcg_re, ln_cache[n, g])

#             print('dbeta_re.shape  = ', dbeta_re.shape)
            print('dgamma_re.shape = ', dgamma_re.shape)

#             out[n, projected_g:projected_g+cg, :, :] = out_re.reshape(cg, x.shape[2], x.shape[3])

#             dbeta[0, projected_g:projected_g+cg, 0, 0] += np.sum(dbeta_re)
#             dgamma[0, projected_g:projected_g+cg, 0, 0] += np.sum(dgamma_re)
    
            xhat[n, projected_g:projected_g+cg, :, :] = ln_cache[n, g][4].reshape(cg, dout.shape[2], dout.shape[3])
        
            dx[n, projected_g:projected_g+cg, :, :] = dx_re.reshape(cg, dout.shape[2], dout.shape[3])    
            

    dbeta = dout.sum(axis=3).sum(axis=2).sum(axis=0).reshape(1, C, 1, 1)    
    
    dgamma = np.sum(dout * xhat, axis=3).sum(axis=2).sum(axis=0).reshape(1, C, 1, 1) 
#     dgamma = dout * 
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx