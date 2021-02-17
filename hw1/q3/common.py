# Common core to Q3
# Author: Andrew Nedea

from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import identity
from scipy.stats import multivariate_normal


# Regularization parameter
ALPHA = 0.00000012

# Maximum number of feature plots (to ease memory concerns)
MAX_FEATURE_PLOTS = 6

@dataclass
class LabeledValue:
    label: object
    value: object


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


def assess_classification(class_data, labeled_data: [LabeledValue], classification: list, priors: dict):
    class_count = len(class_data.keys())

    print('Class count: ', class_count)

    confusion_matrix = np.zeros((class_count, class_count), dtype=float)

    correctly_classified = defaultdict(lambda: 0)
    incorrectly_classified = defaultdict(lambda: 0)

    # Put labels in a sorted list so we can use their indexes to form & access
    # confusion matrix
    labels = sorted(list(class_data.keys()))

    # print('Extracted sorted labels: ', labels)

    for k in range(len(labeled_data)):
        label = labeled_data[k].label

        if label == classification[k]:
            correctly_classified[label] += 1
        else:
            incorrectly_classified[label] += 1

        # Convert classification label -> index
        classification_index = labels.index(classification[k])
        # Convert true label -> index
        label_index = labels.index(label)

        confusion_matrix[classification_index][label_index] += 1

    # print('ccl classified count: ', correctly_classified)
    # print('icc classified count: ', incorrectly_classified)
    
    print('correct classifications: ', sum(list(correctly_classified.values())))
    print('incorrect classifications: ', sum(list(incorrectly_classified.values())))

    # Scale confusion matrix
    for k in range(len(confusion_matrix)):
        for j in range(class_count):
            # Make sure the class has data to it
            if len(class_data[labels[j]]) == 0:
                confusion_matrix[k][j] = 0
            else:
                confusion_matrix[k][j] /= len(class_data[labels[j]]) 
   
    # Initialize var to hold probability of error
    perror = 0
    for j in range(class_count):
        for i in range(class_count):
            # Since we're using a 0-1 loss, this would just be zero
            if j == i:
                continue

            # Make sure we have a prior for this class (if no data has
            # been provided, then there would not be an entry for it in the
            # dict)
            if labels[i] not in priors:
                continue

            # add P(D=j | L=i) * P(L=i)
            perror += confusion_matrix[j][i] * priors[labels[i]]

    print('P(error): ', perror)
    print('Confusion matrix:\n', confusion_matrix)

def render(class_data):
    """
    Renders data three features at a time across classes.
    """
    # For each pair of three features
    for c in range(min(MAX_FEATURE_PLOTS, len(class_data.keys()) - 2)):
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

        feature_count = len(data[class_label][0])

        # Determine mean matrix
        mean_matrices[class_label] = np.mean(data[class_label], axis=0)
        # Determine cov matrix
        cov_matrices[class_label] = np.cov(data[class_label], rowvar=False)

        # Regularize cov matrix
        reg = identity(feature_count) * (np.trace(cov_matrices[class_label]) / 
                np.linalg.matrix_rank(cov_matrices[class_label])) * ALPHA
        cov_matrices[class_label] += reg 

        # Make Gaussian
        gaussians[class_label] = multivariate_normal(mean_matrices[class_label], cov_matrices[class_label])

        # Determine prior
        priors[class_label] = len(data[class_label]) / total_sample_no

        # print(f'class label {class_label} has mean of: ', mean_matrices[class_label])
        # print(f'class label {class_label} has cov of: ', cov_matrices[class_label])

    return priors, gaussians
