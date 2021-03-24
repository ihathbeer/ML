# ML HW3 Question 2
# Author: Andrew Nedea
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
import math
import random
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict

DATA_PATH = 'data/'
IMG_PATH = 'pics/'
CROSSV_K = 10

# Markers & colors for each class label
CLASS_CONFIG = {
    1: {
        'marker': 'o',
        'color1': 'green', 
        'color2': 'orange',
        'color3': 'royalblue',
    },
    2: {
        'marker': 's',
        'color1': 'green',
        'color2': 'red',
        'color3': 'purple', 
    },
}

def load_data(x_path: str, y_path: str) -> ():
    """
    Loads features & corresponding labels from given paths.

    :param x_path: path to features (relative to DATA_PATH)
    :param y_path: path to labels (relative to DATA_PATH)
    :return: tuple of features and labels (X, Y)
    """
    X = genfromtxt(f'{DATA_PATH}/{x_path}', delimiter=',', skip_header=0)
    Y = genfromtxt(f'{DATA_PATH}/{y_path}')

    return X, Y

def find_best_model(X, Y):
    """
    Finds the best parameters sigma & C for an SVM model given
    a dataset. The average error from K-fold cross-validation is used to
    assess a model's performance given parameters sigma & C.

    :param X: features
    :param Y: corresponding labels
    :return: resulting "best" model and a dict that maps (C, sigma) -> error
             for each combination of C and sigma used
    """
    # Create log spaces for parameters
    cspace = np.logspace(-2, 5, num=8) # -2 -> 5, 8
    sigma_space = np.logspace(-2, 6, num=9) # -2 -> 6, 9

    # Initialize results
    min_error = 9999
    best_C = best_sigma = best_gamma = 0

    iteration_no = 1
    total_iteration_no = len(cspace) * len(sigma_space)

    # Dict that maps (C, sigma) -> error
    param_error = {}

    # C-sweep
    for C in cspace:
        # Sigma-sweep
        for sigma in sigma_space:
            # Calculate gamma from sigma
            gamma = 1 / (2 * sigma * sigma)

            # Create SVM model
            model = svm.SVC(kernel='rbf', C=C, gamma=gamma)
            # Perform K-fold cross validation
            scores = cross_val_score(model, X, Y, cv=CROSSV_K, n_jobs=-1)

            # Determine average cross-validation error
            error = 1 - np.mean(scores)

            # Save params and error
            param_error[(C, sigma)] = error

            # See if it beats running best
            if error < min_error:
                min_error = error
                best_sigma = sigma
                best_gamma = gamma
                best_C = C

            if iteration_no % 5 == 0:
                print(f'Iteration {iteration_no}/{total_iteration_no}: C={C} sigma={sigma} E={error}')
            iteration_no += 1

    print(f'Best sigma: {best_sigma}')
    print(f'Best C: {best_C}')
    print(f'Min error: {min_error}')

    # Rebuild model with best params
    best_model = svm.SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    # Train model
    best_model.fit(X, Y)

    return best_model, param_error

def test_model(model, X, Y):
    """
    Calculates the probability of error achieved by the given model
    on the given dataset.

    :param model: model to test
    :param X: features
    :param Y: labels
    :return: error, dict that maps true label to list of correctly classified samples
             and dict that maps true label to list of incorrectly classified samples
    """
    # Predict labels
    predicted_Y = model.predict(X)

    # Keeps track of the no. of incorrect labels
    incorrect_label_no = 0

    cc = defaultdict(list)
    icc = defaultdict(list)

    # Assess classification
    for i in range(len(Y)):
        if predicted_Y[i] != Y[i]:
            incorrect_label_no += 1
            icc[int(Y[i])].append(X[i])
        else:
            cc[int(Y[i])].append(X[i])

    # Calculate error
    error = incorrect_label_no / len(Y)

    print(f'Test error: {error}')

    return error, cc, icc

def plot_classification(ccl: dict, iccl: dict):
    """
    Plots correctly and incorrectly classified samples.

    :param ccl: dict of correctly classified samples that maps label->list of such samples
    :param iccl: dict of incorrectly classified samples that maps label->list of such samples
    :param title: what to name plot
    """
    # Create figure
    fig = plt.figure(1)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'Classification result (testing set)')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    handles = []

    for k in ccl.keys():
        # Convert each entry to an np array for slicing
        ccl[k] = np.array(ccl[k])
        iccl[k] = np.array(iccl[k])

        handles.append(ax.scatter(ccl[k][:, 0], ccl[k][:, 1],
            marker=CLASS_CONFIG[k]['marker'], color=CLASS_CONFIG[k]['color1'],
            label=f'Class {k} correctly classified', s=1.5))

        handles.append(ax.scatter(iccl[k][:, 0], iccl[k][:, 1],
            marker=CLASS_CONFIG[k]['marker'], color=CLASS_CONFIG[k]['color2'],
            label=f'Class {k} incorrectly classified', s=1.5))

    plt.legend(handles=handles)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{IMG_PATH}/classification.png')
    #plt.show()


def plot_data(X: list, Y: list, title: str) -> None:
    """
    Plots given two-feature data and corresponding labels.

    :param X: data to plot (list of tuples)
    :param Y: labels to plot (list of tuples)
    :param title: title of plot
    """
    # Create figure
    fig = plt.figure(int(random.randrange(0, 999999)))
    ax = fig.add_subplot(1, 1, 1)
    # Label plot
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

    # Needed for legend
    handles = []

    # Maps class label to list of X
    samples = defaultdict(list)

    for k in range(len(X)):
        samples[int(Y[k])].append(X[k])

    for k in samples.keys():
        # Convert each entry to an np array for slicing
        samples[k] = np.array(samples[k])

        handles.append(ax.scatter(samples[k][:, 0], samples[k][:, 1],
            marker=CLASS_CONFIG[k]['marker'], color=CLASS_CONFIG[k]['color3'],
            label=f'Class {k}', s=1.5))

    plt.legend(handles=handles)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(f'{IMG_PATH}/{title}.png')
    #plt.show()

def plot_params_vs_error(param_error: dict) -> None:
    """
    Plots parameters C and Sigma on a plot indicating the amount of error at
    each point.

    :param param_error: dict that maps (C, Sigma) -> error
    """
    # Unzip C and Sigma
    X = np.unique([key[0] for key in param_error.keys()])
    Y = np.unique([key[1] for key in param_error.keys()])

    error_matrix = []

    # Build error matrix
    for y in Y:
        row = []
        for x in X:
            row.append(param_error[(x,y)])
        error_matrix.append(row)

    print('X = ', X)
    print('Y = ', Y)
    print('error_matrix: ', error_matrix)

    # Create figure
    fig = plt.figure(0)
    ax = fig.add_subplot(1, 1, 1)
    img = ax.pcolormesh(X, Y, error_matrix, cmap='YlOrRd_r')
    # Label
    ax.set_xlabel('C')
    ax.set_ylabel('Sigma')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Params vs Error')
    # Add colorbar
    cbar = fig.colorbar(img, ax=ax)
    cbar.ax.set_ylabel("Probability of error")
    plt.savefig(f'{IMG_PATH}/params_v_error.png', bbox_inches='tight', pad_inches=0.1)
    #plt.show()

def solve():
    """
    Loads the training and test data, finds the best model parameters using K-fold
    cross-validation on the training data. It then evaluates the "best model" against
    the test dataset.
    """
    train_x, train_y = load_data('train_data.csv', 'train_labels.csv')
    test_x, test_y = load_data('test_data.csv', 'test_labels.csv')

    plot_data(train_x, train_y, 'Training data')
    plot_data(test_x, test_y, 'Testing data')

    print(f'Train X: {train_x.shape} Y: {train_y.shape}')
    print(f'Test X: {test_x.shape} Y: {test_y.shape}')

    best_model, param_error = find_best_model(train_x, train_y)
    # Plot params C and sigma vs error
    plot_params_vs_error(param_error)

    _, cc, icc = test_model(best_model, test_x, test_y)
    plot_classification(cc, icc)

solve()
