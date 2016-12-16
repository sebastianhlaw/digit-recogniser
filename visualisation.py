# Author:   Sebastian Law
# Date:     22-Nov-2016
# Revised:  16-Dec-2016

import numpy as np
from matplotlib.pylab import plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def display_digit(Xi, yi=None):
    # plt.close('all')
    side = np.sqrt(Xi.size).astype(int)
    data = np.array(Xi).reshape((side, side))
    plt.imshow(data, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("y = "+str(yi))
    plt.show()


def plot_first_factors(X_data, y_data, type):
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkgreen', 'darkorange', 'yellow', 'black']
    if type == "pca":
        transform = PCA(n_components=2)
        XX_data = transform.fit(X_data).transform(X_data)
    elif type == "lda":
        transform = LinearDiscriminantAnalysis(n_components=2)
        XX_data = transform.fit(X_data, y_data).transform(X_data)
    else:
        print("Type", type, "not recognised, use either 'pca' or 'lda'.")
        return
    plt.figure()
    for i, j in enumerate(le.classes_):
        plt.scatter(XX_data[y_data == j, 0], XX_data[y_data == j, 1], alpha=0.8, color=colours[i], label=str(j))
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(type)
