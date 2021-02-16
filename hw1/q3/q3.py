# Question 3 - classifying wines
# Author: Andrew Nedea

import numpy as np
from numpy import genfromtxt
import csv
import math
from scipy.stats import multivariate_normal
from scipy.sparse import identity
import matplotlib.pyplot as plt

DATA_PATH = 'data/wine'

# Number of classes (or class labels)
CLASS_COUNT = 10
FEATURE_COUNT = 11

REGULARIZATION_TERM = (identity(FEATURE_COUNT) * 0.05).toarray()

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

        print(f'samples for class {quality_score}: {len(class_data[quality_score])}')

    return class_data

def create_distributions(data: dict) -> tuple:
    """
    Creates Gaussian distribution for each class in dict and
    returns resulting Gaussians & priors.

    :return: tuple(dict(label, prior:int), dict(label, gaussian: scipy.stats.multivariate_normal))
    """
    cov_matrices = {}
    mean_matrices = {}
    priors = {}
    
    gaussians = {}
    pdfs = {}

    total_sample_no = 0
    # Determine total no. of samples across all classes
    for class_label in data:
        total_sample_no += len(data[class_label])

    print('Total sample no.: ', total_sample_no)

    # Cycle over each class, and determine its prior, covariance & mean matrices
    for class_label in data:
        # Skip over classes that do not have any data
        if len(data[class_label]) < 1:
            continue

        # Determine cov matrix
        cov_matrices[class_label] = np.cov(data[class_label], rowvar=False)
        # Regularize cov matrix
        cov_matrices[class_label] += REGULARIZATION_TERM
        # Determine mean matrix
        mean_matrices[class_label] = np.mean(data[class_label], axis=0)
        # Determine prior
        priors[class_label] = len(data[class_label]) / total_sample_no

        print(f' class label {class_label} has prior of: ', priors[class_label])
        # print(f'class label {class_label} has mean of: ', mean_matrices[class_label].shape)
        # print(f'class label {class_label} has cov of: ', cov_matrices[class_label].shape)
        
        gaussians[class_label] = multivariate_normal(mean_matrices[class_label],
                cov_matrices[class_label])

        #pdfs[class_label] = gaussians[class_label].pdf(
    return priors, gaussians

def render(class_data):
    """
    Renders data three features at a time across classes.
    """
    # For each pair of three features
    for c in range(FEATURE_COUNT - 2):
        handles = []

        # Create figure
        fig = plt.figure(c)
        ax1 = fig.add_subplot(1, 1, 1, projection='3d')
        # Set title
        ax1.title.set_text(f'Features {c}, {c+1} and {c+2}')
        
        # For each class, add features
        for class_label in class_data:
            data = class_data[class_label]
            handles.append((ax1.scatter(data[:,c], data[:,c+1], data[:, c+2], label=f'Class {class_label}')))

        # Create legend
        plt.legend(handles=handles)

    # Display windows
    plt.show()

class_data = read_data('winequality-white.csv')
# render(class_data)
priors, gaussians = create_distributions(class_data)

#classify(priors, gaussians)
