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

def plot_decision_boundary(correctly_classified: [(float, float)], incorrectly_classified:\
        [(float, float)], title):
    # make up correctly classified array where index = class label
    cc = [np.array([b.value for b in correctly_classified if b.label == 0]),\
            np.array([b.value for b in correctly_classified if b.label == 1])]

    # make up incorrectly classified array where index = class label
    icc = [np.array([b.value for b in incorrectly_classified if b.label == 0]),\
    np.array([b.value for b in incorrectly_classified if b.label == 1])]

    legend = ['o', '^']
    handles = []

    fig = plt.figure(random.randrange(0, 999999))
    ax0 = fig.add_subplot(1, 1, 1)
    ax0.title.set_text(f'Classification result: {title}')
    ax0.set_xlabel('x0')
    ax0.set_ylabel('x1')

    for k in range(len(cc)):
        handles.append(ax0.scatter(cc[k][:,0], cc[k][:,1], marker=legend[k], color='green',\
                label=f'Class {k} correctly classified'))
        handles.append(ax0.scatter(icc[k][:,0], icc[k][:,1], marker=legend[k], color='red',\
                label=f'Class {k} incorrectly classified'))

    plt.legend(handles=handles)
    #plt.show()
