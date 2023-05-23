import numpy as np

def softmax(x):
    """Return the softmax of a vector x.
    
    :type x: ndarray
    :param x: vector input
    
    :returns: ndarray of same length as x
    """
    x = x - np.max(x)
    row_sum = np.sum(np.exp(x))
    return np.array([np.exp(x_i) / row_sum for x_i in x])


def jacobian_softmax(s):
    """Return the Jacobian matrix of softmax vector s.

    :type s: ndarray
    :param s: vector input

    :returns: ndarray of shape (len(s), len(s))
    """
    return np.diag(s) - np.outer(s, s)

def cross_entropy(y, s):
    """Return the cross-entropy of vectors y and s.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: scalar cost
    """
    # Naively computes log(s_i) even when y_i = 0
    # return -y.dot(np.log(s))
    
    # Efficient, but assumes y is one-hot
    return -np.log(s[np.where(y)])

def gradient_cross_entropy(y, s):
    """Return the gradient of cross-entropy of vectors y and s.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: ndarray of size len(s)
    """
    return -y / s

def error_softmax_input(y, s):
    """Return the sensitivity of cross-entropy cost to input of softmax.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: ndarray of size len(s)
    """
    return s - y

def batch_softmax(x):
    """Return matrix of row-wise softmax of x.

    :type x: ndarray
    :param x: row per example and column per feature

    :returns: ndarray of x.shape after row-wise softmax
    """
    # Stabilize by subtracting row max from each row
    row_maxes = np.max(x, axis=1)
    row_maxes = row_maxes[:, np.newaxis]  # for broadcasting
    x = x - row_maxes

    return np.array([softmax(row) for row in x])

def jacobian_batch_softmax(s):
    """Return array of row-wise Jacobians of s.

    :type s: ndarray
    :param s: matrix whose rows are softmaxed

    :returns: ndarray of shape
              (s.shape[0], s.shape[0], s.shape[1], s.shape[1])
    """
    # Array of nonzero Jacobians lying along tensor diagonal
    return np.array([jacobian_softmax(row) for row in s])

def mean_cross_entropy(y, s):
    """Return the mean row-wise cross-entropy of y and s.

    :type y: ndarray
    :param y: matrix whose rows are one-hot vectors encoding
              the correct class of each example.

    :type s: ndarray
    :param s: matrix whose every row is a softmax distribution over
              class predictions for a given example.

    :returns: scalar, mean row-wise cross-entropy cost
    """
    return np.mean([cross_entropy(y_row, s_row)
                    for y_row, s_row in zip(y, s)])

def jacobian_mean_cross_entropy(y, s):
    """Return the Jacobian matrix for mean cross-entropy.

    :type y: ndarray
    :param y: matrix whose rows are one-hot vectors encoding
              the correct class of each example.

    :type s: ndarray
    :param s: matrix whose every row is a softmax distribution over
              class predictions for a given example.

    :returns: ndarray of shape y.shape holding gradients as rows
    """
    return -(1 / y.shape[0]) * (y / s)

def batch_error_softmax_input(y, s):
    """Return the sensitivity of cross-entropy cost to input of softmax.

    :type y: ndarray
    :param y: matrix whose rows are one-hot vectors encoding
              the correct class of each example.

    :type s: ndarray
    :param s: matrix whose every row is a softmax distribution over
              class predictions for a given example.

    :returns: ndarray of shape y.shape
    """
    return (1 / y.shape[0]) * (S - Y)
