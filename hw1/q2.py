# Question 2
# Author: Andrew Nedea

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class LabeledBox:
    value: object
    label: str

## PRIORS ##
PRIOR1=0.3
PRIOR2=0.3
PRIOR3=0.4

# LOSS MATRICES
ZERO_ONE_LOSS_MA = np.array(
            [[0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]]
        )

TEN_LOSS_MA = np.array(
            [[0, 1, 10],
            [1, 0, 10],
            [1, 1, 0]]
        )

HUNDRED_LOSS_MA = np.array(
            [[0, 1, 100],
            [1, 0, 100],
            [1, 1, 0]]
        )

PRIORS=[PRIOR1, PRIOR2, PRIOR3]

# Total no. of samples
SAMPLE_NO=10000
# Initial sample no. for each distribution
SAMPLE_NO_1=0 # samples for distribution 1
SAMPLE_NO_2=0 # samples for distribution 2
SAMPLE_NO_3=0 # samples for distribution 3
SAMPLE_NO_4=0 # samples for distribution 4

# determine no. of samples per distribution based on priors considering
# that the last two classes share prior3 with equal weight
for k in range(SAMPLE_NO):
    r = random.random()
    if r < PRIOR1:
        SAMPLE_NO_1 += 1
    elif r >= PRIOR1 and r < PRIOR1+PRIOR2:
        SAMPLE_NO_2 += 1
    elif r >= PRIOR1+PRIOR2 and r < PRIOR1+PRIOR2+PRIOR3/2:
        SAMPLE_NO_3 += 1
    else:
        SAMPLE_NO_4 += 1

print('SAMPLE_NO_1:', SAMPLE_NO_1)
print('SAMPLE_NO_2:', SAMPLE_NO_2)
print('SAMPLE_NO_3:', SAMPLE_NO_3)
print('SAMPLE_NO_4:', SAMPLE_NO_4)

print('Total:', SAMPLE_NO_1+SAMPLE_NO_2+SAMPLE_NO_3+SAMPLE_NO_4)
# -------------
# total = 10,000 samples


# ======== CLASS 1 ========
# covariance matrix for class 1
C1 = np.array([
    [4,      0.2,    -0.3],
    [0.2,    4,      0.5],
    [-0.3,     0.5,     1]])

# mean for class 1
M1 = np.array([
    0.2,
    0.2,
    0.2
    ])

# ======== CLASS 2 ========
# covariance matrix for class 2
C2 = np.array([
    [4,       -0.2,    -0.1],
    [-0.2,      9,        0],
    [-0.1,      0,       1]])

# mean for class 2
# for each dimension, it is 2*avg. stddev away from the previous class' mean for that dimension
M2 = np.array([
    4.2, # 0.2 + 2*((sqrt(4)+sqrt(4))/2) = 0.2 + 2*2 = 4.2
    5.2, # 0.2 + 2*((sqrt(4)+sqrt(9))/2) = 0.2 + 5 = 5.2
    2.2  # 0.2 + 2*((sqrt(1)+sqrt(1))/2) = 0.2 + 2*1 = 2.2
    ])

# ======== CLASS 3 =========
# mixture of Gaussian distributions
# first covariance matrix for class 3
C3 = np.array([
    [4,       -0.1,    0.1],
    [-0.1,    4,       0.1],
    [0.1,     0.1,     1]])

# first mean for class 3
M3 = np.array([
    8.2, # 4.2 + 2*((sqrt(4)+sqrt(4))/2)
    10.2, # 5.2 + 2*((sqrt(9)+sqrt(4))/2)
    4.2  # 2.2 + 2*((sqrt(1)+sqrt(1))/2)
    ])

# second covariance matrix for class 3
C4 = np.array([
    [4,       -0.1,    0.1],
    [-0.1,    4,       0.1],
    [0.1,     0.1,     1]])

# second mean for class 3
M4 = np.array([
    12.2, # 8.2 + 2*((sqrt(4)+sqrt(4))/2)
    14.2, # 10.2 + 2*((sqrt(4)+sqrt(4))/2)
    6.2   # 4.2 + 2*((sqrt(1)+sqrt(1))/2)
    ])

def generate_data() -> tuple:
    """
    Generates distributions and samples for each class.

    :return: (distribution 1, samples of X of class 1), (distribution 2, samples of X of class 2),
        (distribution 3, half samples of X of class 3), (distribution 4, half samples 
        of X of class 3)
    """
    # Generate distribution for class 1
    dist1 = multivariate_normal(M1, C1)
    # Sample X|L=1
    xl1 = dist1.rvs(size=SAMPLE_NO_1)
    # Generate distribution for class 2
    dist2 = multivariate_normal(M2, C2)
    # Sample X|L=2
    xl2 = dist2.rvs(size=SAMPLE_NO_2)

    # Generate 1st distribution for class 3
    dist3_0 = multivariate_normal(M3, C3)
    # Sample first half of X|L=3
    xl3_0 = dist3_0.rvs(size=SAMPLE_NO_3)

    # Generate 2nd distribution for class 3
    dist3_1 = multivariate_normal(M4, C4)
    # Sample latter half of X|L=3
    xl3_1 = dist3_1.rvs(size=SAMPLE_NO_4)

    return (dist1, xl1), (dist2, xl2), (dist3_0, xl3_0), (dist3_1, xl3_1)


def plot(xl1, xl2, xl3):
    """
    Plots the X vectors in different colors (to be able to distinguish the classes
    they originally came from).

    :param xl1: X corresponding to the first class
    :param xl2: X corresponding to the second class
    :param xl3: X corresponding to the third class 
    :return: None
    """
    # ========== PLOT =================
    fig = plt.figure(0)
    # xl1 plot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.title.set_text('Samples')
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('x2')
    img1 = ax1.scatter(xl1[:, 0], xl1[:, 1], xl1[:, 2], label='Class 1')
    img2 = ax1.scatter(xl2[:, 0], xl2[:, 1], xl2[:, 2], color='red', label='Class 2')
    img3 = ax1.scatter(xl3[:, 0], xl3[:, 1], xl3[:, 2], color='green', label='Class 3')

    plt.legend(handles=[img1, img2, img3])

def classify(pxl: list, X, loss_matrix: np.array):
    """
    Classifies each X given a loss_matrix and P(L|X) for all L and X.

    :param pxl: MxN array of LabeledBox; M = number of labels; N = number of LabeledBox (samples);
    :param loss_matrix: loss matrix
    :return: list of string decision labels (same length and order as X)
    """
    classification = []

    print('Classifying using loss matrix:', loss_matrix)

    # for each X to classify
    for k in range(len(X)):
        risks = [0, 0, 0]
        # for each decision i
        for i in range(len(risks)):
            # for each label j
            for j in range(len(risks)):
                prob = pxl[j][k].value
                loss = loss_matrix[i][j]
                risks[i] += loss * prob * PRIORS[j]

        # choose action with least risk
        decision = risks.index(min(risks)) + 1
        classification.append(str(decision))

    return classification

def get_pdfs(xl1, xl2, xl3):
    """
    Determines pdf of each given X.

    :param xl1: X corresponding to the first class
    :param xl2: X corresponding to the second class
    :param xl3: X corresponding to the third class 

    :return: returns specified pdfs of the originally given X's as well as a list combining
             all the original X vectors
    """
    # Combine all x's while maintaining labels
    X = list(map(lambda x: LabeledBox(x, '1'), xl1))
    X.extend(list(map(lambda x: LabeledBox(x, '2'), xl2)))
    X.extend(list(map(lambda x: LabeledBox(x, '3'), xl3)))

    # Determine class-conditional probabilities for each sample X
    # p(x|L=1)
    pxl1 = []
    # p(x|L=2)
    pxl2 = []
    # p(x|L=3)
    pxl3 = []

    # Cycle over each and every labeled x
    for lv in X:
        # Determine prob of x given each class
        pxl1.append(LabeledBox(dist1.pdf(lv.value), lv.label))
        pxl2.append(LabeledBox(dist2.pdf(lv.value), lv.label))
        # the pdf of a mixture of two Gaussians is the weighted average
        # of the individual pdf's
        pdf3 = dist3_0.pdf(lv.value) * 0.5 + dist3_1.pdf(lv.value) * 0.5
        pxl3.append(LabeledBox(pdf3, lv.label))

    return pxl1, pxl2, pxl3, X

def solve(xl1, xl2, xl3, loss_matrix: np.array):
    """
    Determines correctly and incorrectly classified X's as well as confusion
    matrix for given parameter.

    :return:  dict indexed by string label (i.e. '1', '2', '3') representing class
              with each index corresponding to a list of vectors x that were classified
              correctly for their class,

              dict indexed by string label (i.e. '1', '2', '3') representing class
              with each index corresponding to a list of vectors x that were classified
              correctly for their class,

              (3x3) confusion matrix
    """
    # Get PDF of each sample set for each class
    pxl1, pxl2, pxl3, X = get_pdfs(xl1, xl2, xl3)
    # Classify
    classification = classify([pxl1, pxl2, pxl3], X, loss_matrix)
    print('classification no.: ', len(classification))

    # Initialize dict to hold correctly classified
    ccl = defaultdict(list)
    # Initialize dict to hold incorrectly classified
    iccl = defaultdict(list)

    # Initialize confusion matrix
    confusion_matrix = np.zeros((3,3), dtype=float)

    # Determine correctly/incorrectly classified x
    for k in range(len(X)):
        label = X[k].label

        # check if L=D (label=L, classification[k]=D)
        if label == classification[k]:
            ccl[label].append(X[k].value)
        else:
            iccl[label].append(X[k].value)
        confusion_matrix[int(classification[k])-1][int(label)-1] += 1

    # Scale confusion matrix
    for k in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[k])):
            class_no_samples = 1
            if j == 0:
                class_no_samples = SAMPLE_NO_1
            elif j == 1:
                class_no_samples = SAMPLE_NO_2
            else:
                class_no_samples = SAMPLE_NO_3 + SAMPLE_NO_4

            confusion_matrix[k][j] = confusion_matrix[k][j] / class_no_samples

    print('Confusion matrix:', confusion_matrix)

    # Convert lists to np.array
    ccl = {k: np.array(v) for k, v in ccl.items()}
    iccl = {k: np.array(v) for k, v in iccl.items()}

    return ccl, iccl, confusion_matrix

def plot_classification(ccl: dict, iccl: dict):
    """
    Plots correctly and incorrectly classified X vectors for each class.

    :param ccl:  dict indexed by string label (i.e. '1', '2', '3') representing class
                 with each index corresponding to a list of vectors X that were classified
                 correctly for their class
    :param iccl: same as above, expect that the vectors X were classified incorrectly for 
                 their class
    :return: None
    """

    # ========== PLOT =================
    fig = plt.figure(1)
    # xl1 plot
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.title.set_text('Classification result')
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('x2')

    # class label L -> marker
    legend = {'1':'o', '2':'^', '3':'s'}

    # initialize container to store scatter handles
    handles = []

    # render each X according to legend with the correctly classified ones rendered in green
    # and the incorrectly classified ones rendered in red
    for k, marker in legend.items():
        handles.append(ax1.scatter(ccl[k][:, 0], ccl[k][:, 1], ccl[k][:, 2], marker=marker,
            color='green', label=f'Class {k} correctly classified'))
        handles.append(ax1.scatter(iccl[k][:, 0], iccl[k][:, 1], iccl[k][:, 2], marker=marker,
            color='red', label=f'Class {k} incorrectly classified'))

    plt.legend(handles=handles)
    plt.show()

if __name__ == "__main__":
    (dist1, xl1), (dist2, xl2), (dist3_0, xl3_0), (dist3_1, xl3_1) = generate_data()

    # Join samples from the last two distributions for class 3
    xl3 = np.concatenate((xl3_0, xl3_1), axis=0)

    plot(xl1, xl2, xl3)

    # Part A (zero-one loss)
    print('Part A. Using Zero one loss')
    ccl, iccl, confusion_matrix = solve(xl1, xl2, xl3, ZERO_ONE_LOSS_MA)
    plot_classification(ccl, iccl)

    # Part B (ten-times loss)
    print('Part B.1 . Using ten times loss')
    ccl, iccl, confusion_matrix = solve(xl1, xl2, xl3, TEN_LOSS_MA)
    plot_classification(ccl, iccl)

    # Part B (hundred-times loss)
    print('Part B.2 . Using hundred times loss')
    ccl, iccl, confusion_matrix = solve(xl1, xl2, xl3, HUNDRED_LOSS_MA)
    plot_classification(ccl, iccl)
