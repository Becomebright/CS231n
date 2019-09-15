import numpy as np
from matplotlib import pyplot as plt
from past.builtins import xrange

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in xrange(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
# lets visualize the data:
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

# Parameter initialization
h = 100
W = 0.01 * np.random.randn(D, h)
b = np.zeros((1, h))
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

# Hyperparameters
step_size = 1e-0
reg = 1e-3

num_examples = X.shape[0]
for i in range(10000):
    # Forward pass
    # hidden_layer = np.dot(X, W) + b
    # hidden_layer_relu = np.maximum(0, hidden_layer)
    # scores = np.dot(hidden_layer_relu, W2) + b2
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W**2) + 0.5 * reg * np.sum(W2**2)
    loss = data_loss + reg_loss

    # Output
    if i % 1000 == 0:
        print("iteration %d / 200: loss %f" % (i, loss))

    # Backpropagation
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples
    dW2 = np.dot(hidden_layer.T, dscores)
    # dW2 = np.dot(hidden_layer_relu.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)
    dhidden = np.dot(dscores, W2.T)
    dhidden[hidden_layer <= 0] = 0
    dW = np.dot(X.T, dhidden)
    db = np.sum(dhidden, axis=0, keepdims=True)
    dW += reg * W
    dW2 += reg * W2

    # Parameter update
    W -= step_size * dW
    b -= step_size * db
    W2 -= step_size * dW2
    b2 -= step_size * db2

scores = np.dot(np.maximum(0, np.dot(X, W) + b), W2) + b2
predicted_class = np.argmax(scores, axis=1)
print('training accuracy: %.2f' % (np.mean(predicted_class == y)))
