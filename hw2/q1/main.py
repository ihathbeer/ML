# ML HW2 Q1
# Author: Andrew Nedea

import numpy as np
from numpy import genfromtxt
from numpy.linalg import pinv, inv
import math
import matplotlib.pyplot as plt
from hw2q1 import hw2q1, plot3, set_aspect_equal_3d, generateDataFromGMM
from sklearn.preprocessing import PolynomialFeatures

SEPARATOR = ''.join(['=']*20)
# y = c(x,w) + v = w0 + x*w1 + x^2*w2 + x^3*w3 + v

#  v ~ N(0, sigma^2)
# MAP: w ~ N(0, gamma * I)

# phi_i(x) = x^i

def phi_i(X, deg):
    """
    Returns the polynomial features of a vector.

    :param X: vector in question
    :param deg: degree of polynomial
    """
    # consider adding interactions
    phi = [1]

    for p in range(1, deg+1):
        for x in X:
            phi.append(math.pow(x, p))
    return phi

def genPolyFeatureMatrix(data, deg):
    """
    Retrieves polynomial feature matrix.

    :param data: matrix containing a series of input vectors X
    :param deg: degree of polynomial
    """
    result = []
    for X in data:
        result.append(phi_i(X, deg))
    return np.array(result)

def trainMAP(train_x: np.array, train_y: np.array, lam: float):
    """
    Computes the weight matrix & squared mean error using MAP estimator.

    :param x_path: input training set
    :param y_path: output training set
    :param lam: regularization term lambda (scalar)
    """
    # determine feature matrix
    featureMatrix = genPolyFeatureMatrix(train_x, 3)

    # calculate weights
    w = (np.transpose(featureMatrix) @ featureMatrix) + np.identity(featureMatrix.shape[1]) * lam
    w = inv(w) @ np.transpose(featureMatrix) @ train_y

    # calculate training error
    error = calculate_error(train_x, train_y, w)

    return w, error

def trainMLE(train_x: np.array, train_y: np.array):
    """
    Computes the weight matrix & squared mean error using MLE estimator.

    :param train_x: input training set
    :param train_y: output training set
    """
    # determine feature matrix
    featureMatrix = genPolyFeatureMatrix(train_x, 3)

    # calculate ML weights
    w_ml = pinv(featureMatrix)@train_y

    # calculate training error
    error = calculate_error(train_x, train_y, w_ml)

    return w_ml, error

def calculate_error(x: np.array, y: np.array, w: np.array) -> float:
    """
    Calculates & returns mean squared error.

    :param x: input set
    :param y: output set
    :param w: weight vector
    """
    # calculate mean squared error
    error = 0

    # make sure we got training sets of equal length
    if(len(x) != len(y)):
        raise Exception('The training sets X and Y must be equal in length!')

    # make prediction & compute error
    for k in range(len(x)):
        prediction = w @ phi_i(x[k], 3)
        actual = y[k]

        error += math.pow(actual - prediction, 2)

    return error / len(x)

def test(validation_x, validation_y, weights):
    """
    Computes the squared mean error for the validation / testing set.

    :param x_path: input validation set
    :param y_path: output validation set
    :param weights: weight matrix
    """
    return calculate_error(validation_x, validation_y, weights)


def mle(training_x, training_y, validation_x, validation_y):
    """
    Performs ML and returns best training & validation errors.

    :param training_x: input dataset to train on
    :param training_y: output dataset to train on
    :param validation_x: input dataset to test on
    :param validation_y: output dataset to test on
    """
    print(SEPARATOR, ' MLE ', SEPARATOR)
    weights, training_error = trainMLE(training_x, training_y)

    print('Weights: ', weights)
    print('Training error: ' , training_error)

    validation_error = test(validation_x, validation_y, weights)

    print('Validation error: ', validation_error)

    return training_error, validation_error

def map(train_x, train_y, validation_x, validation_y, noise):
    """
    Performs map, plots error vs gamma and returns best error.

    :param train_x: input dataset to train on
    :param train_y: output dataset to train on
    :param validation_x: input dataset to test on
    :param validation_y: output dataset to test on
    :param noise: observation noise (beta^-1)
    """
    print(SEPARATOR, ' MAP ', SEPARATOR)
    gamma_space = np.logspace(-4, 4, 200)
    #print('gamma_space:', gamma_space)

    # initialize empty containers to hold plot data for gamma vs squared mean error
    x = []
    y = []

    #  variables to hold optimal gamma & error
    min_error = 9999999
    optimal_gamma = -1
    optimal_weights = []

    for gamma in gamma_space:
        weights, training_error = trainMAP(train_x, train_y, noise/gamma)

        validation_error = test(validation_x, validation_y, weights)

        # print('Validation error: ', validation_error)
        x.append(gamma)
        y.append(validation_error)

        if validation_error < min_error:
            min_error = validation_error
            optimal_gamma = gamma
            optimal_weights = weights

    print('Min error: ', min_error)
    print('Optimal gamma: ', optimal_gamma)
    print('Optimal weights: ', optimal_weights)

    # plot gamma vs squared mean error
    plt.figure(0)
    plt.title('Squared mean error vs Gamma (MAP)')
    plt.xlabel('Gamma')
    plt.ylabel('Error')
    plt.xscale('log')
    plt.plot(np.array(x), np.array(y))
    plt.show()


# generate training & validation data via tool provided by TA
train_x, train_y, validation_x, validation_y = [np.transpose(m) for m in hw2q1()]

beta_inv, validation_error = mle(train_x, train_y, validation_x, validation_y)
map(train_x, train_y, validation_x, validation_y, beta_inv)
