# HW 4 Problem 2
# Author: Andrew Nedea
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import random

RESOURCES_PATH = 'resources/'
IMG_PATH = 'img/'

IMAGES = [
    'airplane',
    'bird'
]

EXTENSION = '.jpg'

MAX_M = 10

HEURISTIC_ELBOW_DELTA = 100000

def read_image(name: str):
    """
    Reads image by name into a list of RGB tuples.

    :param name: image name
    :return: resulting list of RGB (3x1) tuples
    """
    img = image.imread(f'{RESOURCES_PATH}/{name}{EXTENSION}')
    print(f'Read image shape: {img.shape}')
    return img

def process_image(name: str):
    """
    Processed image by name by creating a vector where
    entry is a 5x1 feature vector made up of row no, column no, r, g, b
    of that pixel.

    :param name: name of picture to process'
    :return: resulting vector of feature vectors, width, height
    """
    # Get rgb array for image
    img = read_image(name)

    # Vector to hold feature vector of each pixel
    X = []

    for i in range(len(img)):
        for j in range(len(img[i])):
            r, g, b = img[i][j]
            X.append(np.array([i, j, r, g, b]))

    X = np.array(X)
    # Normalize each feature
    X = normalize(X, norm='max', axis=0)

    return X, len(img), len(img[0])

def plot_gmm(image, component_no: int, name: str):
    """
    Creates and plots a gaussian mixture model for given image with the given number
    of components.

    :param image: image to cluster
    :param component_no: number of clusters (components)
    :return: None
    """
    # Tokenize image entry
    X, width, height = image
    # Create GMM & fit data
    gmm = GaussianMixture(component_no).fit(X)
    # Get clusters (uses MAP behind the scenes)
    prediction = gmm.predict(X)
    # Render clusters
    plt.imshow(prediction.reshape(width, height))
    plt.xlabel('')
    plt.ylabel('')
    plt.title(f'{name} {component_no} components')

    # Save figure
    name = name.replace(' ', '_')
    plt.savefig(f'{IMG_PATH}/{name}_{component_no}_components{EXTENSION}', bbox_inches='tight', pad_inches=0.4)       
    #plt.show()
    plt.clf()

def bic(X, M):
    """
    Calculates the objective function for samples X and number of
    components M.

    :param X: input samples
    :param M: number of components
    """
    # Create GMM & fit data
    gmm = GaussianMixture(M).fit(X)
    # Number of features of each sample
    d = X.shape[1]
    print(f'Avg log-likelihood = ', gmm.score(X))
    # print('Stock BIC = ', gmm.bic(X))
    # Calculate n
    n = (M - 1) + d * M + M * (d * (d + 1) / 2)
    # Calculate pdf of X
    X_pdf = 0

    for m in range(M):
        X_pdf += multivariate_normal.pdf(X, mean=gmm.means_[m,:], cov=gmm.covariances_[m,:,:])*\
                gmm.weights_[m]

    X_pdf = sum(np.log(X_pdf))
    # Calculate & return BIC
    bic = -2 * X_pdf + n*np.log(d * X.shape[0])

    return bic

def plot_metric_vs_component_no(component_no: [float], metric: [float],
        metric_name: str, plot_name: str):
    """
    Plots a metric vs. number of components of a GMM.
    :param component_no: list of no. of component_no
    :param metric: list of metric values
    :param metric_name: name of metric (y)
    :param plot_name: name of plot (title)
    """
    fig = plt.figure(int(random.randrange(0, 999999)))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(component_no, metric)
    ax.set_xlabel('M')
    ax.set_ylabel(metric_name)
    ax.title.set_text(plot_name)
    #plt.show()

    # Format plot_name for hard storage
    plot_name = plot_name.replace(' ', '_')
    plt.savefig(f'{IMG_PATH}/{plot_name}{EXTENSION}', bbox_inches='tight', pad_inches=0.2)       

def find_best_M(image):
    """
    Finds best no. of components for given image by inspecting the BIC
    score and its gradient. If the difference between two consecutive gradients
    falls below a threshold (HEURISTIC_ELBOW_DELTA) then the previous no. of components
    is used.

    :param image: (image matrix, width, height) tuple
    :return: best M (float), ordered list of BIC values for each M (list of float) and 
             ordered list of gradients of BIC values (list of float)
    """
    # Tokenize image entry
    X, width, height = image

    best_bic = 99999999
    reached_elbow = False

    # Containers to store Ms and corresponding BICs
    bics = []
    ms = range(1, MAX_M+1)

    # Try out different no. of components
    for M in ms:
        custom_bic = bic(X, M)
        bics.append(custom_bic)
        print(f'BIC (M = {M}) calculated to: ', custom_bic)

    bic_gradients = np.gradient(bics)

    # Initialize best M
    best_M = 1

    # Determine best M based on gradient
    for m in range(2, MAX_M+1):
        delta = abs(bic_gradients[m-2] - bic_gradients[m - 1])

        if abs(delta) > HEURISTIC_ELBOW_DELTA:
            best_M = m
        else:
            break

    return best_M, bics, bic_gradients

def solve():
    """
    Does a 2-component GMM for each image and then use BIC to determine the optimal number
    of components for each image.
    """
    # Array to hold tuples of (np.array, width, height) for each image
    images = []

    # Process each image
    for img in IMAGES:
        images.append(process_image(img))

    # Do a two-component GMM for each image
    for k in range(len(images)):
        img = images[k]
        plot_gmm(img, 2, IMAGES[k])

    # Create list of Ms to be tried
    all_M = list(range(1, MAX_M+1))

    # Determine best no. of components for each image
    for k in range(len(images)):
        image = images[k]
        best_M, bics, bic_gradients = find_best_M(image)
        print('best_M: ', best_M)

        # Plot BIC vs Ms
        plot_metric_vs_component_no(all_M, bics, 'BIC', f'{IMAGES[k]} BIC vs M')
        plot_metric_vs_component_no(all_M, bic_gradients, 'gradient of BIC',\
                f'{IMAGES[k]} BIC gradient vs M')

        # Plot image with best_M components
        plot_gmm(image, best_M, f'{IMAGES[k]}_bestM')

solve()
