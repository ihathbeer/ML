# Question 3 - classifying wines
# Author: Andrew Nedea

import numpy as np
from numpy import genfromtxt
import csv
from collections import defaultdict
from common import render, create_distributions, classify, assess_classification, render_pca
from sklearn.preprocessing import normalize


DATA_PATH = 'data/wine'

# Number of classes (or class labels)
CLASS_COUNT = 10
FEATURE_COUNT = 11

def read_data(name: str) -> dict:
    """
    Reads in data from csv file and compartmentalizes it
    by class label into dict.

    :return: dict indexed by class label (last column in csv) with each
             value equal to a list of features corresponding to the label.
    """
    data = genfromtxt(f'{DATA_PATH}/{name}', delimiter=';')
    # Drop off first row (it's all headers)
    data = data[1:]
    # Initialize dict to map class label to feature vector
    class_data = {}

    for quality_score in range(0, CLASS_COUNT + 1):
        # Save class specific data
        class_data[quality_score] = data[np.where(data[:,-1] == quality_score)]
        # Drop off last column (class label)
        class_data[quality_score] = class_data[quality_score][:, :-1]

        # Normalize
        if len(class_data[quality_score]) > 0:
            class_data[quality_score] = normalize(class_data[quality_score], axis=0, norm='max')

        print(f'samples for class {quality_score}: {len(class_data[quality_score])}')

    return class_data


class_data = read_data('winequality-red.csv')
render(class_data)
priors, gaussians, cov = create_distributions(class_data)

render_pca(class_data, cov)

labeled_data, classifications = classify(class_data, gaussians, priors)

assess_classification(class_data, labeled_data, classifications, priors)
