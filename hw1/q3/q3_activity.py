# Question 3 - classifying activities
# Author: Andrew Nedea

import numpy as np
from collections import defaultdict
from common import render, create_distributions, classify, assess_classification

DATA_PATH = 'data/activity'

# Number of classes (or class labels)
CLASS_COUNT = 6
FEATURE_COUNT = 561

# Regularization parameter
ALPHA = 0.00000012


def read_data(data_name: str, labels_name: str) -> dict:
    """
    Reads in data from txt file and compartmentalizes it
    by class label into dict.

    :return: dict indexed by class label (last column in txt) with each
             value equal to a list of samples corresponding to the label. Each
             sample is likely to have multiple features.
    """
    data = np.loadtxt(f'{DATA_PATH}/{data_name}')
    labels = np.loadtxt(f'{DATA_PATH}/{labels_name}')

    # Initialize dict to map class label to feature vector
    class_data = defaultdict(list)

    for k in range(len(labels)):
        # Save class specific data
        class_data[int(labels[k])].append(data[k])
    
    for k in class_data:
        class_data[k] = np.array(class_data[k])

    return class_data


class_data = read_data('x_train.txt', 'Y_train.txt')
render(class_data)

priors, gaussians = create_distributions(class_data)

labeled_data, classifications = classify(class_data, gaussians, priors)

assess_classification(class_data, labeled_data, classifications, priors)

