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

def getProbAndError(ratios: list[LabeledBox], gamma: float):
    """
    Determines the probability of a true positive, false positive, false
    negative and minimum error for a sorted list of labeled likelihood ratios and a
    gamma.
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
    tp_prob = tp_no / len(xl1)
    fp_prob = fp_no / len(xl0)
    fn_prob = fn_no / len(xl1)
    # calculate error
    error = fp_prob*PRIOR0+fn_prob*PRIOR1

    return tp_prob, fp_prob, fn_prob, error

### CLASS CONDITIONAL PROBABILITIES ###
# Class 0 class-conditional pdf is Gaussian
# mean for class 0
m0 = np.array([-1, 1, -1, 1])

# covariance matrix for class 0
c0 = np.array([[2, -0.5, 0.3, 0], [-0.5, 1, -0.5, 0], [0.3, -0.5, 1, 0], [0, 0, 0, 2]])

# generate samples that fit class conditional distribution of class 0
# p(x|L=0) -> 4D random vector x|L=0
dist0 = multivariate_normal(m0, c0)
xl0 = dist0.rvs(size=SAMPLE_NO)

# Class 1 class-conditional pdf is Gaussian
# mean for class 1
m1 = np.array([1, 1, 1, 1])

# covariance matrix for class 1
c1 = np.array([[1, 0.3, -0.2, 0], [0.3, 2, 0.3, 0], [-0.2, 0.3, 1, 0], [0, 0, 0, 3]])

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
img0 = ax0.scatter(xl0[:, 0], xl0[:, 1], xl0[:, 2], c=xl0[:, 3], cmap=plt.hot())
fig.colorbar(img0)
# xl1 plot
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
img1 = ax1.scatter(xl1[:, 0], xl1[:, 1], xl1[:, 2], c=xl1[:, 3], cmap=plt.hot())
fig.colorbar(img1)
#plt.show()

# ======= HEURISTICALLY DETERMINED GAMMA ======

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
fnp = [] # false negative
perror = 99999
perror_coord = ()

# establish gamma range
gamma_range = [lr.value for lr in likelihood_ratio if lr.value >= 0]

# cycle over each likeihood ratio & use it as gamma
for gamma in gamma_range:
    tp_prob, fp_prob, fn_prob, error = getProbAndError(likelihood_ratio, gamma)

    tpp.append(tp_prob)
    fpp.append(fp_prob)
    fnp.append(fn_prob)

    # check to see if it is lower than running min
    if error < perror:
        perror = error
        perror_coord = (fpp[-1], tpp[-1])

# ======= THEORETICALLY DETERMINED GAMMA ======
theoretical_gamma = PRIOR0/PRIOR1

print('minimum prob of error: ', perror)
plt.figure(1)
plt.title('ROC graph')
plt.xlabel('Prob. of false positive')
plt.ylabel('Prob. of true positive')
plt.plot(fpp, tpp, label='ROC curve')
plt.plot(perror_coord[0], perror_coord[1], 'bo', label='Heuristic min perror')
plt.show()

