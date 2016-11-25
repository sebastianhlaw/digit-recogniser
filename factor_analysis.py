# Author:   Sebastian Law
# Date:     25-Nov-2016
# Revised:  25-Nov-2016

import loader
from sklearn import preprocessing
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

X, y = loader.load_sklearn()

pca = PCA(n_components=2)
X_pca = pca.fit(X).transform(X)  # also fit_transform(X) not quite sure the difference

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X, y).transform(X)

le = preprocessing.LabelEncoder()
le.fit(y)
labels = ["y=" + str(i) for i in le.classes_]
colours = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'darkgreen', 'darkorange', 'yellow', 'black']

plt.figure()
for i, j in enumerate(le.classes_):
    plt.scatter(X_pca[y == j, 0], X_pca[y == j, 1], alpha=0.8, color=colours[i], label=str(j))
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA')

plt.figure()
for i, j in enumerate(le.classes_):
    plt.scatter(X_lda[y == j, 0], X_lda[y == j, 1], alpha=0.8, color=colours[i], label=str(j))
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')

