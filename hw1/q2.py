# Question 2
# Author: Andrew Nedea

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

PRIOR1=0.3
PRIOR2=0.3
PRIOR3=0.3

SAMPLE_NO_1=2500 # per distribution (=> 10000)
SAMPLE_NO_2=3500
SAMPLE_NO_3=2000
SAMPLE_NO_4=2000
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
    4.2  # 0.7 + 2*((sqrt(1)+sqrt(1))/2)
])

# second covariance matrix for class 3
C4 = np.array([
    [4,       -0.1,    0.1],
    [-0.1,    4,        0],
    [0,       0.1,     1]])

# second mean for class 3
M4 = np.array([
    12.2, # 4.2 + 2*((sqrt(4)+sqrt(4))/2)
    12.2, # 4.2 + 2*((sqrt(4)+sqrt(4))/2)
    6.2  # 0.7 + 2*((sqrt(1)+sqrt(1))/2)
])

dist3_0 = multivariate_normal(M3, C3)
xl3_0 = dist3_0.rvs(size=SAMPLE_NO_3)

dist3_1 = multivariate_normal(M4, C4)
xl3_1 = dist3_1.rvs(size=SAMPLE_NO_4)

xl3 = np.concatenate((xl3_0, xl3_1), axis=0)
print('shape of xl3: ', xl3.shape)

# ========== PLOT =================
fig = plt.figure(0)
# xl1 plot
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.title.set_text('samples of X for class 0')
ax1.set_xlabel('x0')
ax1.set_xlabel('x1')
ax1.set_xlabel('x2')
img1 = ax1.scatter(xl1[:, 0], xl1[:, 1], xl1[:, 2])
img2 = ax1.scatter(xl2[:, 0], xl2[:, 1], xl2[:, 2], color='red')
img3 = ax1.scatter(xl3[:, 0], xl3[:, 1], xl3[:, 2], color='green')
plt.show()
