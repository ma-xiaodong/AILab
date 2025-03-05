from sklearn import linear_model
from sklearn import datasets

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pdb

def plot_boundary(pred_func, data, labels):
    x_min, x_max = data[:, 0].min() - 0.5, data[:, 0].max() + 0.5
    y_min, y_max = data[:, 1].min() - 0.5, data[:, 1].max() + 0.5
    h = 0.01

    # generating a meshgrid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # the result
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    plt.contourf(xx, yy, z, cmap = plt.cm.Blues, alpha = 0.2)
    plt.scatter(data[:, 0], data[:, 1], s = 40, c = labels, 
            cmap = plt.cm.nipy_spectral, edgecolors = "Black")
    plt.title("Logistic Regression")
    plt.show()

def calculate_loss(model, X, y):
    num_examples = len(X)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    # feed forward
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)

    exp_scores = np.exp(a2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

    corect_logprobs = -np.log(probs[range(num_examples), y])
    pdb.set_trace()
    data_loss = np.sum(corect_logprobs)

    data_loss += reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1. / num_examples * data_loss 

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']

    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    a2 = np.tanh(z2)

    exp_scores = np.exp(a2)
    probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)
    return np.argmax(probs, axis = 1)

def ANN_model(X, y, nn_hdim):
    num_indim = len(X)
    model = {}

    np.random.seed(0)
    W1 = np.random.randn(input_dim, nn_hdim) / np.sqrt(input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, output_dim))

    num_passes = 20000
    for i in range(0, num_passes):
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis = 1, keepdims = True)

        delta3 = probs
        delta3[range(num_indim), y] -= 1
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis = 0, keepdims = True)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis = 0)

        dW1 += reg_lambda * W1
        dW2 += reg_lambda * W2

        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        if i % 1000 == 0:
            print("Loss after iteration %i: %f" %(i, calculate_loss(model, X, y)))
    return model

if __name__ == "__main__":
    np.random.seed(0)
    X, y = datasets.make_moons(300, noise = 0.25)

    '''
    # classification use linear regression
    logistic_fun = sklearn.linear_model.LogisticRegression(solver = 'lbfgs', multi_class = 'ovr')
    logistic_fun.fit(X, y)
    plot_boundary(lambda x: logistic_fun.predict(x), X, y)
    plt.title("Logistic Regression")
    '''

    # classification use machine models
    input_dim = 2
    output_dim = 2
    epsilon = 0.01
    reg_lambda = 0.01

    hidden_model = ANN_model(X, y, 3)
    plot_boundary(lambda x: predict(hidden_model, x), X, y)
    plt.title("Hidden Layer size 3")

