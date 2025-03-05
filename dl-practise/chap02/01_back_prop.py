import numpy as np

def loss_der(network_y, real_y):
    return network_y - real_y

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def backprop(x, y, weights, biases):
    # initial value of delta of w and b
    delta_w = [np.zeros(w.shape) for w in weights]
    delta_b = [np.zeros(b.shape) for b in biases]

    # feed forward
    activation = x
    activations = [x]
    zs = []

    for w, b in zip(weights, biases):
        z = np.dot(w, activation) + b
        activation = sigmoid(z)
        zs.append(z)
        activations.append(activation)

    # back propagation
    # loss der at the output layer
    delta_L = loss_der(activations[-1], y) * sigmoid_der(zs[-1])
    delta_b[-1] = delta_L
    delta_w[-1] = np.dot(delta_L, activations[-2].transpose())

    # the last but one layer
    delta_l = delta_L
    for i in range(2, num_layers):
        z = zs[-i]
        sp = sigmoid_der(z)
        delta_l = np.dot(weights[-i + 1] .transpose(), delta_l) * sp
        delta_b[-i] = delta_l
        delta_w[-i] = np.dot(delta_l, activations[-i - 1].transpose())

    return activations[-1], delta_w, delta_b

if __name__ == '__main__':
    network_size = [3, 4, 2]
    num_layers = len(network_size)
    biases = [np.random.randn(h, 1) for h in network_size[1 : ]]
    weights = [np.random.randn(i, j) for (i, j) 
            in zip(network_size[1:], network_size[:-1])]

    training_x = np.random.rand(3).reshape(3, 1)
    training_y = np.array([0, 1]).reshape([2, 1])

    learning_rate = 0.02
    learning_steps = 10000
    for i in range(learning_steps):
        output, delta_w, delta_b = \
                backprop(training_x, training_y, weights, biases)
        if i % 200 == 0:
            print("output: {}".format(output.transpose()))
        for j in range(len(weights)):
            weights[j] = weights[j] - learning_rate * delta_w[j]
            biases[j] = biases[j] = learning_rate * delta_b[j]

