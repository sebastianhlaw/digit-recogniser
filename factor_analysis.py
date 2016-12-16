# Author:   Sebastian Law
# Date:     15-Dec-2016
# Revised:  15-Dec-2016

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def pca_components(data, percentile=0.95, plot=False):
    pca = PCA()
    pca.fit(data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    if plot:
        fig = plt.figure()
        plt.grid()
        plt.ylim(0, 1)
        plt.plot(cumulative)
        plt.show()
    for i, cum in enumerate(cumulative):
        if cum > percentile:
            return i
