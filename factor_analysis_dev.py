# Author:   Sebastian Law
# Date:     25-Nov-2016
# Revised:  15-Dec-2016

from sklearn import preprocessing
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

import loader
import factor_analysis

X, y = loader.load_sklearn()

le = preprocessing.LabelEncoder()
le.fit(y)
labels = ["y=" + str(i) for i in le.classes_]
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkgreen', 'darkorange', 'yellow', 'black']

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)  # also fit_transform(X) not quite sure the difference
X_inv = pca.inverse_transform(X_pca)
plt.figure()
for i, j in enumerate(le.classes_):
    plt.scatter(X_pca[y == j, 0], X_pca[y == j, 1], alpha=0.8, color=colours[i], label=str(j))
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')

# lda = LinearDiscriminantAnalysis(n_components=2)
# X_lda = lda.fit(X, y).transform(X)
# plt.figure()
# for i, j in enumerate(le.classes_):
#     plt.scatter(X_lda[y == j, 0], X_lda[y == j, 1], alpha=0.8, color=colours[i], label=str(j))
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA')

