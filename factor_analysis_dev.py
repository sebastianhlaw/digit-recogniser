# Author:   Sebastian Law
# Date:     25-Nov-2016
# Revised:  15-Dec-2016

from sklearn import preprocessing, model_selection
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt

import loader
import factor_analysis
X_data, y_data = loader.load_sklearn()
# X_data, y_data = loader.load_kaggle_public()

# X_train, X_validate, y_train, y_validate = model_selection.train_test_split(
#     X_data, y_data, train_size=0.25, test_size=0.05, random_state=0)


lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X_data, y_data).transform(X_data)
plt.figure()
for i, j in enumerate(le.classes_):
    plt.scatter(X_lda[y_data == j, 0], X_lda[y_data == j, 1], alpha=0.8, color=colours[i], label=str(j))
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')

