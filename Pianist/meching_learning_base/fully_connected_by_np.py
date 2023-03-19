import numpy as np
import os
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

nn_architecture = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

def init_layers(nn_architecture, seed=99):
    np.random.seed(seed)
    numbers_of_layers = len(nn_architecture)
    # parameters storage initiation
    params_values = {}

    # iteration over per layer
    for idx, layer in enumerate(nn_architecture):
        # we want to start from 1
        layer_index = idx + 1

        # extracting the number of the units in layer
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]

        # initiating the values of the W matrix and vector b for subsequent layers
        params_values['W' + str(layer_index)] = np.random.rand(layer_output_size,
                                                               layer_input_size) * 0.1
        params_values['b' + str(layer_index)] = np.random.rand(layer_output_size,
                                                               1) * 0.1

        # or zero the matrix w and x
        # params_values['W' + str(layer_index)] = np.zeros((layer_input_size,
        #                                                   layer_output_size))
        # params_values['b' + str(layer_index)] = np.zeros((layer_output_size, 1))
    return params_values

# activation function
def sigmod(X):
    return 1 / (1 + np.exp(-X))

def sigmod_backward(dA, X):
    return dA * sigmod(X) * (1 - sigmod(X))

def relu(X):
    return np.maximum(0, X)

def relu_backward(dA, X):
    dX = np.array(dA, copy=True)
    dX[X <= 0] = 0
    return dX

def single_layer_forward_propagation(A_pre, W_curr, b_curr, activation="relu"):

    Z_curr = np.dot(W_curr, A_pre) + b_curr

    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmod
    else:
        raise Exception("Non-supported activation function")

    return activation_func(Z_curr), Z_curr

def full_forward_propagation(X, params_values, nn_architecture):
    # create a memory dictionary to store the information
    # needed for backward propagation process
    memory = {}
    A_curr = X

    # iteration step
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        A_pre = A_curr

        activation_function_curr = layer["activation"]

        W_curr = params_values["W" + str(layer_idx)]
        b_curr = params_values["b" + str(layer_idx)]

        A_curr, Z_curr = single_layer_forward_propagation(A_pre, W_curr, b_curr,
                                                          activation_function_curr)

        # store the information needed in backward propagation
        memory['A' + str(idx)] = A_pre
        memory['Z' + str(layer_idx)] = Z_curr

    return A_curr, memory

def get_loss_value(Y_hat, Y):
    # shape of Y & Y_hat is (1, 900)
    number_of_examples = Y_hat.shape[1]
    cost = -1 / number_of_examples * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))

    return np.squeeze(cost)

def convert_prob_into_class(probs):
    probs_hat = np.copy(probs)
    probs_hat[probs_hat > 0.5] = 1
    probs_hat[probs_hat <= 0.5] = 0
    return probs_hat

def get_accuracy_value(Y_hat, Y):
    Y_hat_ = convert_prob_into_class(Y_hat)
    acc = (Y_hat_ == Y).all(axis = 0).mean()
    return acc

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev,
                                      activation="relu"):
    # numbers of exampls
    m = A_prev.shape[1]

    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmod_backward
    else:
        raise Exception("Non-supported activation function")

    # step-1: {dA_curr, Z_curr} -> dZ_curr
    # dA_curr is the upstream gradient
    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    # step-2: {dZ_curr, A_prev} -> dW_curr (derivate of matrix W)
    # dZ_curr -> db_curr (derivate of vector b)
    # {dZ_curr, W_curr} -> dA_prev
    dW_curr = np.dot(dZ_curr, A_prev.T) / m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = np.dot(W_curr.T, dZ_curr)

    return dA_prev, dW_curr, db_curr

# for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
#     print(layer_idx_prev)
#     print(layer)

def full_backward_propagation(Y_hat, Y, memory, params_values, nn_architecture):
    grads_values = {}
    m = Y.shape[1]
    # ensure the same shape of the label vector and the predicted Y
    Y = Y.reshape(Y_hat.shape)

    # initiation of gradient descent algorithm
    dA_prev = -(np.divide(Y, Y_hat) - np.divide(1 - Y, 1 - Y_hat))

    for layer_idx_prev, layer in reversed(list(enumerate(nn_architecture))):
        layer_idx_curr = layer_idx_prev + 1
        activation_func_curr = layer["activation"]

        dA_curr = dA_prev

        A_prev = memory["A" + str(layer_idx_prev)]
        Z_curr = memory["Z" + str(layer_idx_curr)]

        W_curr = params_values["W" + str(layer_idx_curr)]
        b_curr = params_values["b" + str(layer_idx_curr)]

        dA_prev, dW_curr, db_curr = single_layer_backward_propagation(
            dA_curr, W_curr, b_curr, Z_curr, A_prev, activation_func_curr
        )

        grads_values["dW" + str(layer_idx_curr)] = dW_curr
        grads_values["db" + str(layer_idx_curr)] = db_curr

    return grads_values

def update(params_values, grads_values, nn_architecture, learning_rate):
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1

        params_values["W" + str(layer_idx)] -= learning_rate * grads_values["dW" + str(layer_idx)]
        params_values["b" + str(layer_idx)] -= learning_rate * grads_values["db" + str(layer_idx)]

    return params_values

def train(X, Y, nn_architecture, epochs, learning_rate, verbose=True, callback=None):
    params_values = init_layers(nn_architecture, 2)
    # print(params_values)
    cost_history = []
    acc_history = []

    for i in range(epochs):
        # step forward
        Y_hat, cache = full_forward_propagation(X, params_values, nn_architecture)

        if i == 1:
            print("X.shape:", X.shape)
            print("Y_hat.shape:", Y_hat.shape)
            print("Y.shape:", Y.shape)

        cost = get_loss_value(Y_hat, Y)
        cost_history.append(cost)

        acc = get_accuracy_value(Y_hat, Y)
        acc_history.append(acc)

        grads_values = full_backward_propagation(Y_hat, Y, cache, params_values, nn_architecture)
        # updating the model
        params_values = update(params_values, grads_values, nn_architecture, learning_rate)

        if (i % 50 == 0):
            if (verbose):
                print("Iteration: {:05} - cost: {:.5f} - acc: {:.5f}".format(i, cost, acc))
            if (callback is not None):
                callback(i, params_values)
    return params_values

n = 1000
test_size = 0.1

X, y = make_moons(n_samples=n, noise=0.2, random_state=100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# check the information in X
# print("X.shape:", X.shape)
# print("X_train.shape:", X_train.shape)
# print("X_test.shape:", X_test.shape)

# check the information in Y
# print("y_train.shape:", y_train.shape)
# print("y_test.shape:", y_test.shape)
# print(y_train)

def plot_dataset(X, y, plotname, file_name=None, XX=None, YY=None, preds=None, dark=False):
    if (dark):
        plt.style.use("dark_background")
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16, 12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$X_2$")
    plt.title(plotname, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)

    if (XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha=1, cmap=cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5], cmap="Grays", vmin=0, vmax=.6)

    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral, edgecolors="black")

    if (file_name):
        plt.savefig(file_name)
        plt.close()

# plot_dataset(X, y, "dataset")
# plt.show()

# the formal training process
params_values = train(X=np.transpose(X_train), Y=np.transpose(y_train.reshape((y_train.shape[0], 1))),
                      nn_architecture=nn_architecture, epochs=30000, learning_rate=0.001)

# evaluate the trained model
Y_test_hat, _ = full_forward_propagation(np.transpose(X_test), params_values, nn_architecture)
acc_test = get_accuracy_value(Y_test_hat, y_test)
print("The test accuracy is {:.2f}".format(acc_test))








