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

# total no. of samples
SAMPLE_NO=10000
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
    [0.3,    4,      0.5],
    [-0.2,     0.5,     1]])

# mean for class 1
M1 = np.array([
    0.2,
    0.2,
    0.2
])

dist1 = multivariate_normal(M1, C1)
xl1 = dist1.rvs(size=SAMPLE_NO_1)

# ======== CLASS 2 ========
# covariance matrix for class 2
C2 = np.array([
    [4,       -0.2,    -0.1],
    [-0.1,    4,        0],
    [0,       0.12,     1]])

# mean for class 2
# for each dimension, it is 2*avg. stddev away from the previous class' mean for that dimension
M2 = np.array([
    4.2, # 0.2 + 2*((sqrt(4)+sqrt(4))/2) = 0.2 + 2*2 = 4.2
    4.2, # 0.2 + 2*((sqrt(4)+sqrt(4))/2) = 0.2 + 2*2 = 4.2
    2.2  # 0.2 + 2*((sqrt(1)+sqrt(1))/2) = 0.2 + 2*1 = 2.2
])

dist2 = multivariate_normal(M2, C2)
xl2 = dist2.rvs(size=SAMPLE_NO_2)

# ======== CLASS 3 =========
# mixture of Gaussian distributions
# first covariance matrix for class 3
C3 = np.array([
    [4,       -0.1,    0.1],
    [-0.1,    4,        0],
    [0,       0.1,     1]])

# first mean for class 3
M3 = np.array([
    8.2, # 4.2 + 2*((sqrt(4)+sqrt(4))/2)
    8.2, # 4.2 + 2*((sqrt(4)+sqrt(4))/2)
    4.2  # 2.2 + 2*((sqrt(1)+sqrt(1))/2)
])

# second covariance matrix for class 3
C4 = np.array([
    [4,       -0.1,    0.1],
    [-0.1,    4,        0],
    [0,       0.1,     1]])

# second mean for class 3
M4 = np.array([
    12.2, # 8.2 + 2*((sqrt(4)+sqrt(4))/2)
    12.2, # 8.2 + 2*((sqrt(4)+sqrt(4))/2)
    6.2   # 4.2 + 2*((sqrt(1)+sqrt(1))/2)
])

dist3_0 = multivariate_normal(M3, C3)
xl3_0 = dist3_0.rvs(size=SAMPLE_NO_3)

dist3_1 = multivariate_normal(M4, C4)
xl3_1 = dist3_1.rvs(size=SAMPLE_NO_4)

# get samples for class 3
xl3 = np.concatenate((xl3_0, xl3_1), axis=0)
print('shape of xl3: ', xl3.shape)

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

# ========= Prepare dataset ==========

# Combine all x's while maintaining labels
labeled_v = list(map(lambda x: LabeledBox(x, '1'), xl1))
labeled_v.extend(list(map(lambda x: LabeledBox(x, '2'), xl2)))
labeled_v.extend(list(map(lambda x: LabeledBox(x, '3'), xl3)))

# Determine class-conditional probabilities for each sample X
# p(x|L=1)
pxl1 = []
# p(x|L=2)
pxl2 = []
# p(x|L=3)
pxl3 = []

# Cycle over each and every labeled x
for lv in labeled_v:
    # Determine prob of x given each class
    pxl1.append(LabeledBox(dist1.pdf(lv.value), lv.label))
    pxl2.append(LabeledBox(dist2.pdf(lv.value), lv.label))
    # the pdf of a mixture of two Gaussians is the weighted average
    # of the individual pdf's
    pdf3 = dist3_0.pdf(lv.value) * 0.5 + dist3_1.pdf(lv.value) * 0.5
    pxl3.append(LabeledBox(pdf3, lv.label))

classification: list[int] = []

# Determine heuristic classifications
for k in range(len(pxl1)):
    # compute p(x|L=1)*p(L=1)
    p1 = pxl1[k].value * PRIOR1
    # compute p(x|L=2)*p(L=2)
    p2 = pxl2[k].value * PRIOR2
    # compute p(x|L=3)*p(L=3)
    p3 = pxl3[k].value * PRIOR3
    
    if p1 >= p2 and p1 >= p3:
        classification.append('1')
    elif p2 >= p1 and p2 >= p3:
        classification.append('2')
    else:
        classification.append('3')

print('classification no.: ', len(classification))

# Initialize dict to hold correctly classified
ccl = defaultdict(list)
# Initialize dict to hold incorrectly classified
iccl = defaultdict(list)

# Determine correctly/incorrectly classified x
for k in range(len(labeled_v)):
    label = labeled_v[k].label
    if label == classification[k]:
        ccl[label].append(labeled_v[k].value)
    else:
        iccl[label].append(labeled_v[k].value)

# Convert lists to np.array
for label in ccl:
    ccl[label] = np.array(ccl[label])

for label in iccl:
    iccl[label] = np.array(iccl[label])

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
