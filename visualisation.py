# Author:   Sebastian Law
# Date:     22-Nov-2016
# Revised:  16-Dec-2016

import numpy as np
from matplotlib.pylab import plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def display_digit(X_data, y_data=None, i=0):
    """Display the Xi image, optionally with the corresponding yi label."""
    plt.close('all')
    Xi = X_data[i]
    side = np.sqrt(Xi.size).astype(int)
    data = np.array(Xi).reshape((side, side))
    plt.imshow(data, cmap='Greys', interpolation='nearest')
    plt.title("y = "+str(y_data[i]))
    plt.show()


def plot_first_factors(X_data, y_data, analysis_type="lda"):
    """Generate scatterplot of two principal components (pca or lda) of data set."""
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkgreen', 'darkorange', 'yellow', 'black']
    if analysis_type == "pca":
        transform = PCA(n_components=2)
        XX_data = transform.fit(X_data).transform(X_data)
    elif analysis_type == "lda":
        transform = LinearDiscriminantAnalysis(n_components=2)
        XX_data = transform.fit(X_data, y_data).transform(X_data)
    else:
        print("Type", analysis_type, "not recognised, use either 'pca' or 'lda'.")
        return
    plt.figure()
    for i, j in enumerate(le.classes_):
        plt.scatter(XX_data[y_data == j, 0], XX_data[y_data == j, 1], alpha=0.8, color=colours[i], label=str(j))
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title(analysis_type)
