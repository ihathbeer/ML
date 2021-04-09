# HW 4 Problem 2
# Author: Andrew Nedea
import numpy as np
from matplotlib import image
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

IMG_PATH = 'resources/'

IMAGES = {
    'airplane.jpg',
    'bird.jpg'
}

def read_image(name: str):
    """
    Reads image by name into a list of RGB tuples.

    :param name: image name
    :return: resulting list of RGB (3x1) tuples
    """
    img = image.imread(f'{IMG_PATH}/{name}')
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

def run_gmm(images, component_no: int):
    # Do a gaussian mixture
    for image in images:
        # Tokenize image entry
        X, width, height = image
        # Create GMM & fit data
        gmm = GaussianMixture(component_no).fit(X)
        # Get clusters
        prediction = gmm.predict(X)
        # Render clusters
        plt.imshow(prediction.reshape(width, height))
        #plt.show()

def bic(X, M):
    """
    Calculates the objective function for samples X and number of
    components M.

    :param X: input samples
    :param M: number of components
    """
    # Create GMM & fit data
    gmm = GaussianMixture(M, random_state=0).fit(X)
    print(f'Doing bic for M={M}')
    # Number of features of each sample
    d = X.shape[1]

    #print(f'd = {d}')
    #print(f'S = {X.shape[0]}')
    print(f'score = ', gmm.score(X))
    print('BIC = ', gmm.bic(X))
    # Calculate n
    n = (M - 1) + d * M + M * (d * (d + 1) / 2)
    
    X_pdf = 0

    for m in range(M):
        X_pdf += multivariate_normal.pdf(X, mean=gmm.means_[m,:], cov=gmm.covariances_[m,:,:])*\
                gmm.weights_[m]

    X_pdf = np.log(X_pdf)
    neg_log_likelihood = -2 * sum(X_pdf) 
    # Calculate & return BIC
    bic = neg_log_likelihood + n*np.log(d * X.shape[0])

    return bic

def solve():
    # Array to hold tuples of (np.array, width, height) for each image
    images = []

    # Process each image
    for img in IMAGES:
        images.append(process_image(img))

    # Do a two-component GMM
    run_gmm(images, 2)

    for image in images:
        # Tokenize image entry
        X, width, height = image

        best_bic = -99999999
        best_M = 0

        # Try out different no. of components
        for M in range(1, 12):
            custom_bic = bic(X, M)

            if custom_bic < best_bic:
                best_bic = custom_bic
                best_M = M

            print(f'custom bic (M = {M}) calculated to: ', custom_bic)
        print('best_M: ', best_M)

solve()
