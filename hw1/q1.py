# Author: Andrew Nedea
# Question 1
import numpy as np
import math
from scipy.stats import multivariate_normal
from scipy.linalg import logm, expm
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class LabeledBox:
    value: object
    label: str
        
# Number of samples to generate per distribution
SAMPLE_NO=5000

## PRIORS ###
PRIOR0 = 0.7
PRIOR1 = 0.3

# mean for class 0
M0 = np.array([-1, 1, -1, 1])
# covariance matrix for class 0
C0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])
# diagonal covariance matrix for class 0
C0_DIAG = np.array([
    [2, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 2]])

# mean for class 1
M1 = np.array([1, 1, 1, 1])
# covariance matrix for class 1
C1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])
# diagonal covariance matrix for class 1
C1_DIAG = np.array([
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 3]])

def getProbAndError(ratios: list[LabeledBox], n0: int, n1: int, gamma: float):
    """
    Determines the probability of a true positive, false positive, false
    negative and minimum error for a sorted list of labeled likelihood ratios and a
    gamma. Each likelihood ratio is labeled with the class the x in the ratio P(x|w1)/P(x|w0)
    came from.

    :param ratios: P(x|w1)/P(x|w0)
    :param n0: number of samples X corresponding to class 0
    :param n1: number of samples X corresponding to class 1

    :return: prob of true positive, prob of false positive & prob of min error
    """
    # initialize counters for false pos, true pos, and false negative
    fp_no = 0
    tp_no = 0
    fn_no = 0

    # cycle over each likelihood ratio
    for lr in ratios:
        if lr.label == '0' and lr.value >= gamma:
            # D = 1, L = 0
            fp_no += 1
        elif lr.label == '1':
            if lr.value >= gamma:
                # D = 1, L = 1
                tp_no += 1
            else:
                # D = 0, L = 1
                fn_no += 1

    # append probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1
    # calculate error
    error = fp_prob*PRIOR0+fn_prob*PRIOR1

    return tp_prob, fp_prob, error

def solve(dist0, dist1, xl0: np.array, xl1: np.array):
    """
    Finds the heuristic & theoretical minimum probability of error for ERM
    classification.
    """
    # combine the two xl's while keeping track of what class each belongs to
    labeled_v = list(map(lambda x: LabeledBox(x, '0'), xl0))
    labeled_v.extend(list(map(lambda x: LabeledBox(x, '1'), xl1)))

    # initialize container to hold all p(x|L=1)/p(x|L=0)
    likelihood_ratio = []
    # cycle over each x|L
    for lv in labeled_v:
        likelihood_ratio.append(LabeledBox(dist1.pdf(lv.value)/dist0.pdf(lv.value), lv.label))

    # sort container by value
    likelihood_ratio.sort(key=lambda lv: lv.value)

    # initialize containers to hold probabilities
    tpp = [] # true positive
    fpp = [] # false positive
    best_error = 99999
    best_error_coord = ()
    best_gamma = -1

    # establish gamma range
    gamma_range = [lr.value for lr in likelihood_ratio if lr.value >= 0]

    # cycle over each likelihood ratio & use it as gamma
    for gamma in gamma_range:
        tp_prob, fp_prob, error = getProbAndError(likelihood_ratio, len(xl0), len(xl1), gamma)

        tpp.append(tp_prob)
        fpp.append(fp_prob)

        # check to see if it is lower than running min
        if error < best_error:
            best_error = error
            best_error_coord = (fpp[-1], tpp[-1])
            best_gamma = gamma

    # print('first 15 tpp: ', tpp[:15])
    # print('first 15 fpp: ', fpp[:15])

    # print('last 15 tpp: ', tpp[-15:])
    # print('last 15 fpp: ', fpp[-15:])

    print('Heuristical best error: ', best_error)
    print('Heuristical best gamma: ', best_gamma)

    # ======= THEORETICALLY OPTIMAL DETERMINED GAMMA ======
    to_gamma = PRIOR0/PRIOR1
    to_tp_prob, to_fp_prob, to_error = getProbAndError(likelihood_ratio, len(xl0), len(xl1), to_gamma) 

    print('Theoretically optimal error: ', to_error)
    print('Theoretical best gamma: ', to_gamma)

    plt.figure(1)
    plt.title('ROC graph')
    plt.xlabel('Prob. of false positive')
    plt.ylabel('Prob. of true positive')
    plt.plot(fpp, tpp, label='ROC curve')
    plt.plot(best_error_coord[0], best_error_coord[1], 'bo', label='Heuristic error')
    plt.plot(to_fp_prob, to_tp_prob, 'ro', label='Theoretic error')
    plt.legend()
    plt.show()

def generateAndSampleTwoClasses(m0, c0, m1, c1):
    """
    Creates two Gaussian distributions with given means and covariances.

    :param m0: mean of first distribution
    :param c0: coveriance matrix of first distribution
    :param m1: mean of second distribution
    :param c1: coveriance matrix of second distribution
    :return: [distribution1, distribution2, SAMPLE_NO of samples from distribution1,
             SAMPLE_NO of samples from distribution2]
    """
    # generate samples that fit class conditional distribution of class 0
    # p(x|L=0) -> 4D random vector x|L=0
    dist0 = multivariate_normal(m0, c0)
    xl0 = dist0.rvs(size=SAMPLE_NO)

    # generate samples that fit class conditional distribution of class 0
    # p(x|L=1) -> 4D random vector
    dist1 = multivariate_normal(m1, c1)
    xl1 = dist1.rvs(size=SAMPLE_NO)

    # xl0[:, k] returns kth columns of xl0
    # plot each column one a separate axis, with the exception with the last, which is plotted as a
    # color
    fig = plt.figure(0)
    # xl0 plot
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    ax0.title.set_text('samples of X for class 0')
    ax0.set_xlabel('x0')
    ax0.set_xlabel('x1')
    ax0.set_xlabel('x2')
    img0 = ax0.scatter(xl0[:, 0], xl0[:, 1], xl0[:, 2], c=xl0[:, 3], cmap=plt.hot())
    fig.colorbar(img0)
    # xl1 plot
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    ax1.title.set_text('samples of X for class 1')
    ax1.set_xlabel('x0')
    ax1.set_xlabel('x1')
    ax1.set_xlabel('x2')
    img1 = ax1.scatter(xl1[:, 0], xl1[:, 1], xl1[:, 2], c=xl1[:, 3], cmap=plt.hot())
    fig.colorbar(img1)

    return dist0, dist1, xl0, xl1

def main():
    print('Generating distributions and samples for Part A!')
    dist0, dist1, xl0, xl1 = generateAndSampleTwoClasses(M0, C0, M1, C1)
    solve(dist0, dist1, xl0, xl1)

    print('Generating distributions for Part B!')
    dist0, dist1, _, _ = generateAndSampleTwoClasses(M0, C0_DIAG, M1, C1_DIAG)
    solve(dist0, dist1, xl0, xl1)

main()
