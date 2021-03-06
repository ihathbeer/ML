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


def generateData(n):
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

    return class0_samples, class1_samples, dist01, dist02, dist1, sample_no_01+sample_no_02, sample_no_1

d20_train = generateData(20)
d200_train = generateData(200)
d2000_train = generateData(2000)
d10000_validate = generateData(10000)


@dataclass
class LabeledBox:
    label: int
    value: float

def label_data(data):
    # choose L1 if p(L=1|x) > p(L=0|x)
    #              p(x|L=1)*p(L=1) > p(x|L=0)*p(L=0)
    class0_samples, class1_samples, dist01, dist02, dist1, count0, count1 = data

    # label samples
    class0_samples_labeled = [LabeledBox(0, v) for v in class0_samples]
    class1_samples_labeled = [LabeledBox(1, v) for v in class1_samples]

    # combined labeled samples
    all_samples_labeled = np.concatenate((class0_samples_labeled, class1_samples_labeled), axis=0)

    return all_samples_labeled, dist01, dist02, dist1

def part1_classify(all_samples_labeled, dist01, dist02, dist1, gamma):
    # initialize container to hold classifications
    classifications = []

    for b in all_samples_labeled:
        # for class 0 compute likelihood
        likelihood0 = (0.5 * dist01.pdf(b.value) + 0.5 *  dist02.pdf(b.value))
        # for class 1 compute likelihod
        likelihood1 = dist1.pdf(b.value)

        # push decision
        classifications.append(0 if (likelihood0/likelihood1 > gamma) else 1)

    return classifications

def part1_compute(data, n0, n1, gamma):
    all_samples_labeled, dist01, dist02, dist1 = data
    classifications = part1_classify(all_samples_labeled, dist01, dist02, dist1, gamma)

    # assess classifications
    error = tp_no = tn_no = fp_no = fn_no = 0

    # determine true positive, false positive, true negative and false negative
    for k in range(len(all_samples_labeled)):
        if(classifications[k] == all_samples_labeled[k].label):
            # correctly classified
            if classifications[k] == 1:
                tp_no += 1
            else:
                tn_no +=1
        else:
            # incorrectly classified
            if classifications[k] == 1:
                fp_no += 1
            else:
                fn_no += 1

    # determine probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1

    # determine error
    error = fp_prob * p0 + fn_prob * p1

    return error, tp_prob, fp_prob, fn_prob


def part1():
    generated_data = d10000_validate
    n0, n1 = generated_data[-2:]
    data = label_data(generated_data)

    gamma_range = np.linspace(0, 10, 40)

    x = []
    y = []
    best_error = 99999
    best_error_coord = [-1, -1]

    for gamma in gamma_range:
        error, tpp, fpp, fnp = part1_compute(data, n0, n1, gamma)

        x.append(fpp)
        y.append(tpp)

        if error < best_error:
            best_error = error
            best_error_coord = [fpp, tpp]

        print(f'For gamma = {gamma}, error: = {error}')

    plt.figure(1)
    plt.title('ROC graph')
    plt.xlabel('Prob. of false positive')
    plt.ylabel('Prob. of true positive')
    plt.plot(x, y, label='ROC curve')
    plt.plot(best_error_coord[0], best_error_coord[1], 'bo', label='Heuristic error')
    #plt.plot(to_fp_prob, to_tp_prob, 'ro', label='Theoretic error')
    plt.legend()
    plt.show()
part1()
