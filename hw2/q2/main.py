# ML HW2 Q2
# Author: Andrew Nedea

import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from dataclasses import dataclass
import random

# Prior probabilities
p0 = 0.65
p1 = 0.35

# Weights
w1 = 0.5
w2 = 0.5

# Means
# Class 0 means
m01 = [3, 0]
m02 = [0, 3]
# Class 1 means
m1 = [2, 2]

# Covariance matrices
# Class 0 covariances
c01 = [[2, 0],
        [0, 1]]

c02 = [[1, 0],
        [0, 2]]

# Class 1 covariances
c1 = [[1, 0],
        [0, 1]]


def generate_data(n):
    print(f' ======== Generating {n} samples ======== ')
    # p(x|L=0) = w1 g(x|m01, C01) + w2 * g(x|m02, C02)
    # p(x|L=1) = g(x|m1, C1)

    # initialize vars to keep track of the number of samples in each distribution
    sample_no_01 = 0
    sample_no_02 = 0
    sample_no_1 = 0

    # produce samples according to priors
    for k in range(n):
        v = random.random()

        if v < p0:
            # generate samples for class 0 (mixture)
            if v < p0 * w1:
                # produce sample for first gaussian component
                sample_no_01 += 1
            else:
                # produce sample for second gaussian component
                sample_no_02 += 1
        else:
            # generate samples for class 1
            sample_no_1 += 1

    print('Class 0 sample no.: ', sample_no_01+sample_no_02)
    print(' > component 1 sample no.: ', sample_no_01)
    print(' > component 2 sample no.: ', sample_no_02)
    print('Class 1 sample no.: ', sample_no_1)

    # create distribution for first Gaussian component of class 0
    dist01 = multivariate_normal(m01, c01)
    # create distribution for second Gaussian component of class 0
    dist02 = multivariate_normal(m02, c02)
    # create distribution for class 1
    dist1 = multivariate_normal(m1, c1)

    # get samples for class 0 from its two Gaussian components
    class0_samples = np.concatenate((dist01.rvs(size=sample_no_01), dist02.rvs(size=sample_no_02)),
            axis=0)
    # get samples for class 1
    class1_samples = dist1.rvs(size=sample_no_1)

    return class0_samples, class1_samples, dist01, dist02, dist1, sample_no_01+sample_no_02,\
            sample_no_1

d20_train = generate_data(20)
d200_train = generate_data(200)
d2000_train = generate_data(2000)
d10000_validate = generate_data(10000)


@dataclass
class LabeledBox:
    label: int
    value: float

def label_data(class0_samples: [(int, int)], class1_samples: [(int, int)]):
    """
    Combined and labels given samples as a list of LabeledBox objects.

    :param class0_samples: samples of class 0
    :param class1_samples: samples of class 1
    """
    # label samples
    class0_samples_labeled = [LabeledBox(0, v) for v in class0_samples]
    class1_samples_labeled = [LabeledBox(1, v) for v in class1_samples]

    # combined labeled samples
    all_samples_labeled = np.concatenate((class0_samples_labeled, class1_samples_labeled), axis=0)

    return all_samples_labeled

def part1_classify(ratios: [float], n0: int, n1: int, gamma: float):
    """
    Takes in a list of likelihood ratios, the no. of samples for class 0, the no. of samples
    for class1 and a threshold gamma. Returns the the min probability of error, the
    probability of a true positive, that of a false positive, that of a false negative,
    a list containing the indexes of correctly classified samples & a list of indexes of
    incorrectly classified samples.

    :param ratios: list of ratios
    :param n0: size of sample space for class 0
    :param n1: size of sample space for class 1
    :param gamma: threshold
    """
    # assess classifications
    error = tp_no = tn_no = fp_no = fn_no = 0

    # holds indexes of correctly classified samples
    correctly_classified = []
    # holds indexes of incorrectly classified samples
    incorrectly_classified = []

    for k in range(len(ratios)):
        # make decision
        decision = 1 if (ratios[k].value >= gamma) else 0;

        # assess classification
        if(decision == ratios[k].label):
            # correctly classified
            if decision == 1:
                tp_no += 1
            else:
                tn_no += 1
            correctly_classified.append(k)
        else:
            # incorrectly classified
            if decision == 1:
                fp_no += 1
            else:
                fn_no += 1
            incorrectly_classified.append(k)

    # determine probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1

    # determine error
    error = fp_prob * p0 + fn_prob * p1

    return error, tp_prob, fp_prob, fn_prob, correctly_classified, incorrectly_classified


def part1_solve():
    """
    Computes the ROC curve, heuristic & theoretical error coordinates for a Bayesian threshold
    based classification rule.

    :return: the X coordinates of the ROC curve, the Y coordinates of the ROC curve, the
             coordinates of the best heuristically determined error, the coordinates of the
             best theoretically derived error, a list of correctly classified samples and
             a list of incorrectly classified samples
    """
    # demultiplex generated data
    samples0, samples1, dist01, dist02, dist1, n0, n1 = d10000_validate

    # label data
    all_samples_labeled = label_data(samples0, samples1)
    
    # container to hold all labeled likelihood ratios
    likelihood_ratios = []
    max_likelihood_ratio = -9999

    # calculate likelihood ratios
    for b in all_samples_labeled:
        # for class 0 compute likelihood
        likelihood0 = (0.5 * dist01.pdf(b.value) + 0.5 *  dist02.pdf(b.value))
        # for class 1 compute likelihod
        likelihood1 = dist1.pdf(b.value)

        # calculate ratio
        ratio = likelihood1 / likelihood0
        likelihood_ratios.append(LabeledBox(b.label, ratio))

        # determine max likelihood ratio
        max_likelihood_ratio = max(max_likelihood_ratio, ratio)

    # do a gamma sweep
    gamma_range = np.linspace(0, max_likelihood_ratio, 1000)

    roc_x = []
    roc_y = []

    # heuristic best error
    heuristic_error = 99999
    heuristic_error_coord = (-1, -1)
    heuristic_gamma = -9999

    # for every gamma
    for gamma in gamma_range:
        # compute error, true positive, false positive and false negative count
        error, tpp, fpp, fnp, _, _ = part1_classify(likelihood_ratios, n0, n1, gamma)

        roc_x.append(fpp)
        roc_y.append(tpp)

        # determine best error
        if error < heuristic_error:
            heuristic_error = error
            heuristic_error_coord = [fpp, tpp]
            heuristic_gamma = gamma

    # theoretical best error
    th_best_gamma = p0 / p1
    th_error, th_tpp, th_fpp, _, cci, icci = part1_classify(likelihood_ratios, n0, n1,
            th_best_gamma)

    print('Theoretic best gamma: ', th_best_gamma)
    print('Theoretic best error: ', th_error)
    print('Heuristic best error: ', heuristic_error)
    print('Heuristic best gamma: ', heuristic_gamma)

    # map indices to actual samples of correctly & incorrectly classified samples
    cc = [all_samples_labeled[i] for i in cci]
    icc = [all_samples_labeled[i] for i in icci]

    return roc_x, roc_y, heuristic_error_coord, (th_fpp, th_tpp), cc, icc

def plot_roc_part1(roc_x: list[float], roc_y: list[float], heuristic_error_coord: tuple,
        th_error_coord: tuple):
    # plot ROC
    plt.figure(0)
    plt.title('ROC graph')
    plt.xlabel('Prob. of false positive')
    plt.ylabel('Prob. of true positive')
    plt.plot(roc_x, roc_y, label='ROC curve')
    plt.plot(heuristic_error_coord[0], heuristic_error_coord[1], 'bo', label='Heuristic error')
    plt.plot(th_error_coord[0], th_error_coord[1], 'ro', label='Theoretic error')
    plt.legend()
    # plt.show()

def plot_decision_boundary(correctly_classified: [(float, float)], incorrectly_classified:\
        [(float, float)]):
    # make up correctly classified array where index = class label
    cc = [np.array([b.value for b in correctly_classified if b.label == 0]),\
            np.array([b.value for b in correctly_classified if b.label == 1])]

    # make up incorrectly classified array where index = class label
    icc = [np.array([b.value for b in incorrectly_classified if b.label == 0]),\
    np.array([b.value for b in incorrectly_classified if b.label == 1])]

    legend = ['o', '^']
    handles = []

    fig = plt.figure(1)
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.title.set_text('Classification result using ideal gamma')
    ax0.set_xlabel('x0')
    ax0.set_ylabel('x1')

    for k in range(len(cc)):
        handles.append(ax0.scatter(cc[k][:,0], cc[k][:,1], marker=legend[k], color='green',\
                label=f'Class {k} correctly classified'))
        handles.append(ax0.scatter(icc[k][:,0], icc[k][:,1], marker=legend[k], color='red',\
                label=f'Class {k} incorrectly classified'))

    plt.legend(handles=handles)
    plt.show()

def part1():
    roc_x, roc_y, heuristic_error_coord, th_error_coord, cc, icc = part1_solve()

    plot_roc_part1(roc_x, roc_y, heuristic_error_coord, th_error_coord)
    plot_decision_boundary(cc, icc)

### part 2 ####
def sigmoid(x):
    #print('sigmoid of: ', x)
    return 1/(1+np.exp(-x))

# h(x, w)'s:
def get_linear_features(x):
    return np.transpose([[1]*x.shape[0], x[:,0], x[:,1]])

def get_quadratic_features(x):
    return np.transpose([[1]*x.shape[0], x[:,0], x[:,1], np.power(x[:,0], 2), x[:,0]*x[:,1],
        np.power(x[:,1], 2)])

def extract_xy(dataset):
    samples0, samples1, dist01, dist02, dist1, n0, n1 = dataset
    # label data
    all_samples_labeled = label_data(samples0, samples1)
    # unlabeled samples
    X = np.concatenate((samples0, samples1), axis=0)
    # solely labels
    Y = np.array([b.label for b in all_samples_labeled])

    return X, Y

def logistic_regression(train_x, train_y, test_x, test_y, lr, epochs, quadratic: bool):
    """
    Performs logistic regression on training set & reports results for test set.

    :param train_x: list of 2D training samples X
    :param train_y: list of training labels Y
    :param test_x:  list of 2D testing samples X
    :param test_y: list of testing labels Y
    :param lr: learning rate
    :param epochs: number of iterations to train model
    :param quadratic: whether to use quadratic features of X (defaults to linear)
    """
    # initialize weights
    w = np.zeros((6, 1)) if quadratic else np.zeros((3, 1))
    # initialize size of data set
    n = len(train_x)
    # initialize array to store cost history
    cost_history = []
    # set feature function
    z = get_quadratic_features if quadratic else get_linear_features

    for k in range(epochs):
        y_predicted = sigmoid(w.T @ z(train_x).T)
        # calculate cost
        c = (-1/n) * np.sum(train_y * np.log(y_predicted) + (1-train_y) * np.log(1-y_predicted))

        # print cost every 10k epochs
        if(k % 10000 == 0):
            print(f'[epoch = {k}] cost =', c)

        # determine gradient w.r.t w
        gradient = (1/n) * (z(train_x).T @ (y_predicted - train_y).T)
        # adjust weights
        w = w - lr * gradient
        cost_history.append(c)

    # now test on test data
    predicted_y = sigmoid(w.T @ z(test_x).T).T

    # extract size of class 0 labels (from test)
    n0 = sum(y == 0 for y in test_y)
    # extract size of class 1 labels (from test)
    n1 = sum(y == 1 for y in test_y)

    print('min predicted y: ', min(predicted_y))
    print('max predicted y: ', max(predicted_y))
    predicted_y = [0 if py < 0.5 else 1 for py in predicted_y]
    correctly_classified = []
    incorrectly_classified = []

    tp_no = fp_no = fn_no = tn_no = 0

    for k in range(len(predicted_y)):
        #print('comparing: ', predicted_y[k], ' to ', test_y[k])
        if(predicted_y[k] == test_y[k]):
            # correctly classified
            if predicted_y[k] == 1:
                tp_no += 1
            else:
                tn_no += 1
            correctly_classified.append(LabeledBox(test_y[k], test_x[k]))
        else:
            # incorrectly classified
            if predicted_y[k] == 1:
                fp_no += 1
            else:
                fn_no += 1
            incorrectly_classified.append(LabeledBox(test_y[k], test_x[k]))

    # determine probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1

    print(' - true positive count: ', tp_no)
    print(' - false positive count: ', fp_no)
    print(' - false negative count: ', fn_no)
    # determine error
    error = fp_prob * p0 + fn_prob * p1

    print('error: ', error)

    plot_decision_boundary(correctly_classified, incorrectly_classified)
    
def part2():
    # ==== LOGISTIC LINEAR ======
    # demultiplex generated data
    x20, y20 = extract_xy(d20_train)
    x200, y200 = extract_xy(d200_train)
    x2000, y2000 = extract_xy(d2000_train)
    x10k, y10k = extract_xy(d10000_validate)

    print('\n\n======= LINEAR ========')
    print('---> d20')
    logistic_regression(x20, y20, x10k, y10k, 0.001, 50000, False)
    print('---> d200')
    logistic_regression(x200, y200, x10k, y10k, 0.001, 50000, False)
    print('---> d2000')
    logistic_regression(x2000, y2000, x10k, y10k, 0.001, 50000, False)
    print('\n\n======= QUADRATIC ========')
    print('---> d20')
    logistic_regression(x20, y20, x10k, y10k, 0.005, 50000, True)
    print('---> d200')
    logistic_regression(x200, y200, x10k, y10k, 0.005, 50000, True)
    print('---> d2000')
    logistic_regression(x2000, y2000, x10k, y10k, 0.005, 50000, True)


#part1()
part2()
