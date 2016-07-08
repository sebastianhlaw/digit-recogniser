# Author:   Sebastian Law
# Date:     30-Jun-2016
# Revised:  08-Jul-2016

import numpy as np

import loader



# # Load the digits dataset
# data, labels = loader.load_kaggle()
# n = len(labels)


from sklearn import svm

from sklearn import cross_validation

X, y = loader.load_sklearn()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, y, test_size=0.3, random_state=0)
classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
classifier.score(X_test, y_test)
# y_predict = classifier.predict(X_test)
