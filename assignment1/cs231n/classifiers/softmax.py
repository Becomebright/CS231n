from builtins import range
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

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_class = W.shape[1]
    dS = np.zeros((num_train, num_class))

    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]
        sum_exp_scores = np.sum(np.exp(scores))
        loss = loss - correct_class_score + np.log(sum_exp_scores)
        for j in range(num_class):
            if j == y[i]:
                dS[i, j] = -1 + np.exp(scores[j]) / sum_exp_scores
            else:
                dS[i, j] = np.exp(scores[j]) / sum_exp_scores

    loss /= num_train
    loss += reg * np.sum(W**2)

    dW = X.T.dot(dS) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_scores = scores[np.arange(num_train), y]
    sum_exp = np.sum(np.exp(scores), axis=1)
    loss = -np.sum(correct_scores) + np.sum(np.log(sum_exp))
    loss = loss / num_train + reg * np.sum(W**2)

    dS = np.zeros_like(scores)
    dS = np.exp(scores) / sum_exp.reshape(num_train, 1)
    dS[np.arange(num_train), y] -= 1
    dW = X.T.dot(dS) / num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
