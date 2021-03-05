# ML HW2 Q1
# Author: Andrew Nedea

import numpy as np
from numpy import genfromtxt
from numpy.linalg import pinv, inv
import math
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

def loadData(path: str):
    """
    Loads dataset by path and returns it as as np.array.
    """
    return np.array(genfromtxt(path, delimiter=','))

def trainMAP(x_path: str, y_path: str, gamma: float):
    """
    Computes the weight matrix & squared mean error using MAP estimator.

    :param x_path: path to input training set
    :param y_path: path to output training set
    :param gamma: regularization term (scalar)
    """
    # load training data
    train_x = loadData(x_path)
    train_y = loadData(y_path)

    # determine feature matrix
    featureMatrix = genPolyFeatureMatrix(train_x, 3)
    
    # calculate weights
    w = (np.transpose(featureMatrix) @ featureMatrix) + gamma * np.identity(featureMatrix.shape[1])
    w = inv(w) @ np.transpose(featureMatrix) @ train_y

    # calculate training error
    error = calculate_error(train_x, train_y, w)

    return w, error

def trainMLE(x_path: str, y_path: str):
    """
    Computes the weight matrix & squared mean error using MLE estimator.

    :param x_path: path to input training set
    :param y_path: path to output training set
    """
    # load training data
    train_x = loadData(x_path)
    train_y = loadData(y_path)

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

def test(x_path, y_path, weights):
    """
    Computes the squared mean error for the validation / testing set.

    :param x_path: path to input validation set
    :param y_path: path to output validation set
    :param weights: weight matrix
    """
    # load data
    test_x = loadData(x_path)
    test_y = loadData(y_path)

    return calculate_error(test_x, test_y, weights)


def mle():
    print(SEPARATOR, ' MLE ', SEPARATOR)
    weights, training_error = trainMLE('data/xtrain.csv', 'data/ytrain.csv')

    print('Weights: ', weights)
    print('Training error: ' , training_error)

    validation_error = test('data/xvalidate.csv', 'data/yvalidate.csv', weights)

    print('Validation error: ', validation_error)

def map():
    print(SEPARATOR, ' MAP ', SEPARATOR)
    gamma_space = np.linspace(math.pow(10, -4), math.pow(10, 4), 1000)

    for gamma in gamma_space:
        print('For gamma: ', gamma)
        weights, training_error = trainMAP('data/xtrain.csv', 'data/ytrain.csv', gamma)

        validation_error = test('data/xvalidate.csv', 'data/yvalidate.csv', weights)

        print('Validation error: ', validation_error)

        # print('Weights: ', weights)
        # print('Training error: ' , training_error)

mle()
map()
