from hw2q1 import hw2q1
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
from keras import models
from keras import layers
from sklearn.model_selection import cross_val_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import tensorflow as tf
import random
import math
import logging
import random
from sklearn.metrics import mean_squared_error

# Max. no. of perceptrons to try out for ANN MLP
MAX_PERCEPTRON_NO = 30
FEATURE_NO = 2
EPOCH_NO = 200
K_SPLIT = 10

# Disable TF warnings (temp. measure)
tf.get_logger().setLevel(logging.ERROR)
# Seed pseudo-random no. generator
random.seed()


def create_nn(perceptron_no: int):
    """
    Creates a multi-layer perceptrons with the specified number of perceptrons
    for its first layer. Its second layer uses a softplus activation function.

    :param perceptron_no: number of perceptrons
    :return: compiled model
    """
    net = models.Sequential()
    # add hidden layer
    net.add(layers.Dense(units=perceptron_no, activation='softplus', input_shape=(FEATURE_NO,)))
    # add output layer
    net.add(layers.Dense(units=1))
    # compile
    net.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])

    return net

def create_regressor(perceptron_no: int, epoch_no: int=500):
    """
    Creates and returns regressor based on a MLP with given number of perceptrons
    and trained given number of times.

    :param perceptron_no: number of perceptrons
    :param epoch_no: number of epochs to train ANN MLP
    :return: KerasRegressor object configured to spec
    """
    nn = KerasRegressor(build_fn=create_nn, perceptron_no=perceptron_no, epochs=epoch_no,
            batch_size=100, verbose=0)

    return nn

def run_cross_val(X, Y, perceptron_no, epoch_no):
    """
    Runs k-fold cross-validation on given training data.
    :param X: data
    :param Y: labels
    :param perceptron_no: number of perceptrons of NN
    :param epoch_no: number of epochs to train on
    :return: average mean squared error obtained from the k-folds
    """
    print(f'Sample no: {len(Y)} {len(X)}')

    estimator = create_regressor(perceptron_no, epoch_no)

    scores = cross_val_score(estimator, X, Y, cv=K_SPLIT, scoring='neg_mean_squared_error')

    return abs(np.mean(scores))

def find_best_config(X, Y):
    """
    Finds the best number of perceptron using k-fold cross-validation.
    :param X: training datapoints
    :param Y: training labels
    :return: best model traing on whole training set
    """
    # maps set size -> optimal perceptron no.
    optimal_perceptron_no = -1

    if len(X) != len(Y):
        raise ValueError('No. of samples must equal no. of labels!')

    # Run cross-validation to determine best config (no. of perceptrons)
    print(f'=== Training set of size {len(X)} ===')
    best_error = 9999
    best_perceptron_no = 0

    perceptron_no_to_error = defaultdict(list)

    # try out different no. of perceptrons
    for k in range(1, MAX_PERCEPTRON_NO + 1):
        # get mean squared error from cross-validation
        mse = run_cross_val(X, Y, k, EPOCH_NO)

        # save the error for this config
        perceptron_no_to_error[k] = mse
        print(f'Perceptron no. = {k} Cross-validation MSE: {mse}')

        # check if this configuration beats the running best error-wise
        if mse < best_error:
            best_error = mse
            best_perceptron_no = k

    print(f'Optimal no. of perceptrons: {best_perceptron_no}')
    print(f'Best cross-validation MSE: {best_error}')
    print('\n')

    plot_error_vs_perceptron_no(perceptron_no_to_error)
    # Reconstruct optimal model & train it on (whole) training set
    # Rebuild model with same no. of perceptrons it achieved best error on
    best_model = create_regressor(best_perceptron_no)
    # Train model
    best_model.fit(X, Y)

    return best_model

def plot_error_vs_perceptron_no(perceptron_no_to_error):
    """
    Plots the error achieved by each configuration (no. of perceptrons).
    """
    # Configure layout
    count = 1

    perceptron_no = list(perceptron_no_to_error.keys())
    error = list(perceptron_no_to_error.values())

    fig = plt.figure(999)
    # Create plot
    ax = fig.add_subplot(1, 1, 1)
    ax.title.set_text(f'Cross-validation avg MSE vs perceptron no.')
    ax.set_xlabel('No. of perceptrons')
    ax.set_ylabel('Average mean-squared error')
    ax.stem(perceptron_no, error, '-.')
    plt.show()

def plot_solution(test_x, test_y: [float], predicted_y: [float]):
    """
    Plots actual vs predicted data.

    :param test_x: test inputs
    :param test_y: actual test outputs
    :param predicted_y: predicted test outputs
    """
    fig = plt.figure()
    # setup plot
    ax1 = fig.add_subplot(111, projection='3d')
    handles = []
    # add test data scatter
    handles.append(ax1.scatter(test_x[:,1], test_x[:,0], test_y, color='g',
        label='Test set w/ actual labels'))
    handles.append(ax1.scatter(test_x[:,1], test_x[:,0], predicted_y, color='r',
        label='Test set w/ predicted labels'))

    # add prediction scatter
    ax1.set_xlabel("x2")
    ax1.set_ylabel("x1")
    ax1.set_zlabel("y")
    ax1.title.set_text('Results')
    plt.legend(handles=handles)
    plt.show()

def solve(train_x, train_y, test_x, test_y):
    """
    Solves problem by finding best number of perceptrons based on the
    mean-squared-error obtained from performing 10 fold cross-validation
    on the training set. The model is configured with the best no. of 
    perceptrons, trained on the training dataset and assessed on the test
    dataset.

    :param train_x: training inputs
    :param train_y: training outputs
    :param test_x: test inputs
    :param test_y: test outputs
    """
    best_model = find_best_config(train_x, train_y)

    # predict using best model
    predicted_y = best_model.predict(test_x)
    # determine MSE on test data set
    test_mse = mean_squared_error(test_y, predicted_y)

    print(f'Test MSE: {test_mse}')
    
    # plot solution
    plot_solution(test_x, test_y, predicted_y)


train_x, train_y, test_x, test_y = [np.transpose(m) for m in hw2q1()]

solve(train_x, train_y, test_x, test_y)
