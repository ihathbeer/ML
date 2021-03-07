import common
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def get_linear_features(x):
    return np.transpose([[1]*x.shape[0], x[:,0], x[:,1]])

def get_quadratic_features(x):
    return np.transpose([[1]*x.shape[0], x[:,0], x[:,1], np.power(x[:,0], 2), x[:,0]*x[:,1],
        np.power(x[:,1], 2)])

def extract_xy(dataset):
    samples0, samples1, dist01, dist02, dist1, n0, n1 = dataset
    # label data
    all_samples_labeled = common.label_data(samples0, samples1)
    # unlabeled samples
    X = np.concatenate((samples0, samples1), axis=0)
    # solely labels
    Y = np.array([b.label for b in all_samples_labeled])

    return X, Y

def logistic_regression(train_x, train_y, test_x, test_y, lr, epochs, quadratic: bool):
    """
    Performs logistic regression on training set & reports results for test set.

    :param train_x: list of 2D training samples X
    :param train_y: list of training labels Y
    :param test_x:  list of 2D testing samples X
    :param test_y: list of testing labels Y
    :param lr: learning rate
    :param epochs: number of iterations to train model
    :param quadratic: whether to use quadratic features of X (defaults to linear)
    """
    # initialize weights
    w = np.zeros((6, 1)) if quadratic else np.zeros((3, 1))
    # initialize size of data set
    n = len(train_x)
    # initialize array to store cost history
    cost_history = []
    # set feature function
    z = get_quadratic_features if quadratic else get_linear_features

    for k in range(epochs):
        y_predicted = sigmoid(w.T @ z(train_x).T)
        # calculate cost
        c = (-1/n) * np.sum(train_y * np.log(y_predicted) + (1-train_y) * np.log(1-y_predicted))

        # print cost every 10k epochs
        if(k % 10000 == 0):
            print(f'[epoch = {k}] cost =', c)

        # determine gradient w.r.t w
        gradient = (1/n) * (z(train_x).T @ (y_predicted - train_y).T)
        # adjust weights
        w = w - lr * gradient
        cost_history.append(c)

    # now test on test data
    predicted_y = sigmoid(w.T @ z(test_x).T).T

    # extract size of class 0 labels (from test)
    n0 = sum(y == 0 for y in test_y)
    # extract size of class 1 labels (from test)
    n1 = sum(y == 1 for y in test_y)

    # establish classification rule
    predicted_y = [0 if py < 0.5 else 1 for py in predicted_y]

    # assess classification using testing set
    correctly_classified = []
    incorrectly_classified = []

    tp_no = fp_no = fn_no = tn_no = 0

    for k in range(len(predicted_y)):
        if(predicted_y[k] == test_y[k]):
            # correctly classified
            if predicted_y[k] == 1:
                tp_no += 1
            else:
                tn_no += 1
            correctly_classified.append(common.LabeledBox(test_y[k], test_x[k]))
        else:
            # incorrectly classified
            if predicted_y[k] == 1:
                fp_no += 1
            else:
                fn_no += 1
            incorrectly_classified.append(common.LabeledBox(test_y[k], test_x[k]))

    # determine probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1

    print(' - true positive count: ', tp_no)
    print(' - false positive count: ', fp_no)
    print(' - false negative count: ', fn_no)

    # determine error
    error = fp_prob * common.p0 + fn_prob * common.p1

    print('error: ', error)

    title = f'quadratic logistic [{n}]' if quadratic else f'linear logistic [{n}]'
    common.plot_decision_boundary(correctly_classified, incorrectly_classified, title)
    
def execute(d20_train, d200_train, d2000_train, d10000_validate):
    # ==== LOGISTIC LINEAR ======
    # demultiplex generated data
    x20, y20 = extract_xy(d20_train)
    x200, y200 = extract_xy(d200_train)
    x2000, y2000 = extract_xy(d2000_train)
    x10k, y10k = extract_xy(d10000_validate)

    print('\n\n======= LINEAR ========')
    print('---> d20')
    logistic_regression(x20, y20, x10k, y10k, 0.01, 50000, False)
    print('---> d200')
    logistic_regression(x200, y200, x10k, y10k, 0.03, 50000, False)
    print('---> d2000')
    logistic_regression(x2000, y2000, x10k, y10k, 0.03, 50000, False)
    print('\n\n======= QUADRATIC ========')
    print('---> d20')
    logistic_regression(x20, y20, x10k, y10k, 0.005, 50000, True)
    print('---> d200')
    logistic_regression(x200, y200, x10k, y10k, 0.005, 50000, True)
    print('---> d2000')
    logistic_regression(x2000, y2000, x10k, y10k, 0.005, 50000, True)

    plt.show()
