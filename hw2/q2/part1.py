import common
import numpy as np
import matplotlib.pyplot as plt


def classify(ratios: [float], n0: int, n1: int, gamma: float):
    """
    Takes in a list of likelihood ratios, the no. of samples for class 0, the no. of samples
    for class1 and a threshold gamma. Returns the the min probability of error, the
    probability of a true positive, that of a false positive, that of a false negative,
    a list containing the indexes of correctly classified samples & a list of indexes of
    incorrectly classified samples.

    :param ratios: list of ratios
    :param n0: size of sample space for class 0
    :param n1: size of sample space for class 1
    :param gamma: threshold
    """
    # assess classifications
    error = tp_no = tn_no = fp_no = fn_no = 0

    # holds indexes of correctly classified samples
    correctly_classified = []
    # holds indexes of incorrectly classified samples
    incorrectly_classified = []

    for k in range(len(ratios)):
        # make decision
        decision = 1 if (ratios[k].value >= gamma) else 0;

        # assess classification
        if(decision == ratios[k].label):
            # correctly classified
            if decision == 1:
                tp_no += 1
            else:
                tn_no += 1
            correctly_classified.append(k)
        else:
            # incorrectly classified
            if decision == 1:
                fp_no += 1
            else:
                fn_no += 1
            incorrectly_classified.append(k)

    # determine probabilities
    tp_prob = tp_no / n1
    fp_prob = fp_no / n0
    fn_prob = fn_no / n1

    # determine error
    error = fp_prob * common.p0 + fn_prob * common.p1

    return error, tp_prob, fp_prob, fn_prob, correctly_classified, incorrectly_classified


def solve(dataset):
    """
    Computes the ROC curve, heuristic & theoretical error coordinates for a Bayesian threshold
    based classification rule.

    :return: the X coordinates of the ROC curve, the Y coordinates of the ROC curve, the
             coordinates of the best heuristically determined error, the coordinates of the
             best theoretically derived error, a list of correctly classified samples and
             a list of incorrectly classified samples
    """
    # demultiplex generated data
    samples0, samples1, dist01, dist02, dist1, n0, n1 = dataset

    # label data
    all_samples_labeled = common.label_data(samples0, samples1)
    
    # container to hold all labeled likelihood ratios
    likelihood_ratios = []
    max_likelihood_ratio = -9999

    # calculate likelihood ratios
    for b in all_samples_labeled:
        # for class 0 compute likelihood
        likelihood0 = (0.5 * dist01.pdf(b.value) + 0.5 *  dist02.pdf(b.value))
        # for class 1 compute likelihod
        likelihood1 = dist1.pdf(b.value)

        # calculate ratio
        ratio = likelihood1 / likelihood0
        likelihood_ratios.append(common.LabeledBox(b.label, ratio))

        # determine max likelihood ratio
        max_likelihood_ratio = max(max_likelihood_ratio, ratio)

    # do a gamma sweep
    gamma_range = np.linspace(0, max_likelihood_ratio, 1000)

    roc_x = []
    roc_y = []

    # heuristic best error
    heuristic_error = 99999
    heuristic_error_coord = (-1, -1)
    heuristic_gamma = -9999

    # for every gamma
    for gamma in gamma_range:
        # compute error, true positive, false positive and false negative count
        error, tpp, fpp, fnp, _, _ = classify(likelihood_ratios, n0, n1, gamma)

        roc_x.append(fpp)
        roc_y.append(tpp)

        # determine best error
        if error < heuristic_error:
            heuristic_error = error
            heuristic_error_coord = [fpp, tpp]
            heuristic_gamma = gamma

    # theoretical best error
    th_best_gamma = common.p0 / common.p1
    th_error, th_tpp, th_fpp, _, cci, icci = classify(likelihood_ratios, n0, n1,
            th_best_gamma)

    print('Theoretic best gamma: ', th_best_gamma)
    print('Theoretic best error: ', th_error)
    print('Heuristic best error: ', heuristic_error)
    print('Heuristic best gamma: ', heuristic_gamma)

    # map indices to actual samples of correctly & incorrectly classified samples
    cc = [all_samples_labeled[i] for i in cci]
    icc = [all_samples_labeled[i] for i in icci]

    return roc_x, roc_y, heuristic_error_coord, (th_fpp, th_tpp), cc, icc

def plot_roc(roc_x: list[float], roc_y: list[float], heuristic_error_coord: tuple,
        th_error_coord: tuple):
    # plot ROC
    plt.figure(0)
    plt.title('ROC graph')
    plt.xlabel('Prob. of false positive')
    plt.ylabel('Prob. of true positive')
    plt.plot(roc_x, roc_y, label='ROC curve')
    plt.plot(heuristic_error_coord[0], heuristic_error_coord[1], 'bo', label='Heuristic error')
    plt.plot(th_error_coord[0], th_error_coord[1], 'ro', label='Theoretic error')
    plt.legend()
    # plt.show()

def execute(dataset):
    roc_x, roc_y, heuristic_error_coord, th_error_coord, cc, icc = solve(dataset)

    plot_roc(roc_x, roc_y, heuristic_error_coord, th_error_coord)
    common.plot_decision_boundary(cc, icc, 'using ideal lambda')
