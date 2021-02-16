# Question 3 - classifying activities
# Author: Andrew Nedea

import numpy as np
import sys
from numpy import genfromtxt
import csv
import math
from scipy.stats import multivariate_normal
from scipy.sparse import identity
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from sklearn.covariance import ShrunkCovariance
from sklearn.datasets import make_gaussian_quantiles

DATA_PATH = 'data/activity'

# Number of classes (or class labels)
CLASS_COUNT = 6
FEATURE_COUNT = 561

# Regularization parameter
ALPHA = 0.00000012


@dataclass
class LabeledValue:
    label: object
    value: object


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

        # Determine mean matrix
        mean_matrices[class_label] = np.mean(data[class_label], axis=0)
        # Determine cov matrix
        cov_matrices[class_label] = np.cov(data[class_label], rowvar=False)

        # Regularize cov matrix
        reg = identity(FEATURE_COUNT) * (np.trace(cov_matrices[class_label]) / 
                np.linalg.matrix_rank(cov_matrices[class_label])) * ALPHA
        cov_matrices[class_label] += reg 

        # Make Gaussian
        gaussians[class_label] = multivariate_normal(mean_matrices[class_label], cov_matrices[class_label])

        # Determine prior
        priors[class_label] = len(data[class_label]) / total_sample_no

        # print(f'class label {class_label} has mean of: ', mean_matrices[class_label])
        print(f'class label {class_label} has cov of: ', cov_matrices[class_label])

    return priors, gaussians

def render(class_data):
    """
    Renders data three features at a time across classes.
    """
    # For each pair of three features
    for c in range(3):
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

def classify(class_data: dict, gaussians, priors: dict):
    # Initialize container to store all the data
    all_data = []

    # Combine data across classes into container
    for class_label in class_data:
        for sample in class_data[class_label]:
            all_data.append(LabeledValue(class_label, sample))

    # Classify
    classification = []

    # For each data sample
    for sample in all_data:
        best_p = 0
        best_label = -1

        # Determine P(x|L)*P(L) of this sample for every class L
        for class_label in gaussians:
            # P(x|L)
            likelihood = gaussians[class_label].pdf(sample.value)
            # P(L)
            prior = priors[class_label]
            # Compute class-conditioned likelihood * prior
            p = likelihood * prior

            # Determine if resulting prior*likelihood beats running best
            if p > best_p:
                best_p = p
                best_label = class_label

        classification.append(best_label)

    return all_data, classification
    
def assess_classification(class_data, labeled_data: [LabeledValue], classification: list):
    confusion_matrix = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=float)

    correctly_classified = defaultdict(lambda: 0)
    incorrectly_classified = defaultdict(lambda: 0)

    for k in range(len(labeled_data)):
        label = labeled_data[k].label

        if label == classification[k]:
            correctly_classified[label] += 1
        else:
            incorrectly_classified[label] += 1

        confusion_matrix[int(classification[k])-1][int(label)-1] += 1

    print('ccl classified count: ', correctly_classified)
    print('icc classified count: ', incorrectly_classified)
    
    print('correct classifications: ', sum(list(correctly_classified.values())))
    print('incorrect classifications: ', sum(list(incorrectly_classified.values())))

    # Scale confusion matrix
    for k in range(len(confusion_matrix)):
        for j in range(CLASS_COUNT):
            # Make sure the class has data to it
            if len(class_data[j]) == 0:
                confusion_matrix[k][j] = 0
            else:
                confusion_matrix[k][j] /= len(class_data[j]) 
   
    #print('Confusion matrix:', confusion_matrix)


class_data = read_data('x_train.txt', 'Y_train.txt')
#render(class_data)

priors, gaussians = create_distributions(class_data)

labeled_data, classifications = classify(class_data, gaussians, priors)

assess_classification(class_data, labeled_data, classifications)

