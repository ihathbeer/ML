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

part1()
