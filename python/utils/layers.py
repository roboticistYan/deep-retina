import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = x.reshape(x.shape[0], -1).dot(w) + b
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

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx = dout.dot(w.T).reshape(x.shape)
  dw = x.reshape(x.shape[0], -1).T.dot(dout)
  db = np.sum(dout, axis=0)
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
  out = np.maximum(0, x)
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
  x = cache
  dx = np.where(x > 0, dout, 0)
  return dx


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
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    mask = (np.random.rand(*x.shape) < p) / p
    out  = x * mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  elif mode == 'test':
    out = x

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

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
  if mode == 'train':
    dx = dout * mask

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  elif mode == 'test':
    dx = dout
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
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
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

def mse_loss(x, y):
  """
  Computes the mean squared error loss and gradient for logistic regression.
  Assumes that y is a vector of probabilities (for instance, of firing).

  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of ground truth, of shape (N,) where y[i] is the desired ith output.

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx:   Gradient of the loss with respect to x
  """
  # compute logistic regression forward pass

  # guarantee that y matches probs
  loss   = np.mean((x - y)**2)

  # backward pass
  dx = 2*(x - y)
  dx /= x.shape[0]

  return loss, dx

def cross_entropy_loss(x, y):
  """
  Computes the cross entropy or logistic loss.
  Assumes that y is a vector of probabilities (for instance, of firing), and x is a vector of scores.

  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of ground truth, of shape (N,) where y[i] is the desired ith output.

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx:   Gradient of the loss with respect to x
  """
  # if you don't do this step, loss is shape (num_ex, num_ex)
  #desqueezeX = False
  #if len(x.shape) > 1:
  #    x = x.squeeze()
  #    desqueezeX = True
  #if len(y.shape) > 1:
  #    y = y.squeeze()

  # only compute cross_entropy_loss when y or (1-y) are not zero
  aboveZero = (y > 0)
  belowOne  = ((1. - y) > 0)

  rates = x.copy() # don't want to alter the original x's
  rates[rates<=0.0] = 10e-4
  rates[rates>=1.0] = 1. - 10e-4

  losses = np.zeros(rates.shape)
  losses[aboveZero] -= y[aboveZero] * np.log(rates[aboveZero])
  losses[belowOne]  -= (1. - y[belowOne]) * np.log(1. - rates[belowOne])
  loss = np.mean(losses)
    
  # backward pass
  dx = -y/rates + (1.-y)/(1.-rates)
  dx /= rates.shape[0]
    

  if (np.isnan(dx)).any():
      import pdb
      pdb.set_trace()

  #if desqueezeX:
  #    dx = np.expand_dims(dx, 1)

  return loss, dx

def mqe_loss(x, y):
  """
  Computes the mean squared error loss and gradient for logistic regression.
  Assumes that y is a vector of probabilities (for instance, of firing).

  Inputs:
  - x: Input data, of shape (N,) where x[i] is the score for the ith input.
  - y: Vector of ground truth, of shape (N,) where y[i] is the desired ith output.

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx:   Gradient of the loss with respect to x
  """
  # compute logistic regression forward pass

  # guarantee that y matches probs
  loss   = np.mean((x - y)**4)

  # backward pass
  dx = 4*((x - y)**3)
  dx /= x.shape[0]
  return loss, dx

def logistic_forward(x, params):
  """
  Computes the logistic function forward pass.
  """
  a, b, c = params
  cache   = (x, params)
  out     = a / (1 + np.exp(-b * (x - c)))

  return out, cache

def logistic_backward(dout, cache):
  """
  Computes the logistic function backward pass.
  """
  x, params = cache

  a, b, c = params
  y = a / (1 + np.exp(-b * (x - c)))
  
  dx = y * (1 - y) * dout
  dalpha  = 1. / (1 + np.exp(-b * (x - c)))
  dgain   = - a * (c - x) * np.exp(-b * (x - c)) / ((1 + np.exp(-b * (x - c)))**2)
  dthresh = - a * b * np.exp(-b * (x - c)) / ((1 + np.exp(-b * (x - c)))**2)
  dparams = np.array([dalpha.T.dot(dout), dgain.T.dot(dout), dthresh.T.dot(dout)]).squeeze()

  return dx, dparams


  

