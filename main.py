# Author:   Sebastian Law
# Date:     30-Jun-2016
# Revised:  30-Jun-2016

# import os
# import pandas as pd
# import numpy as np
#
# def get_data():
#     data_path = os.path.join(os.path.expanduser('~'), 'Google Drive', 'Datasets', 'Kaggle', 'Digit Recognizer')
#     train = pd.read_csv(os.path.join(data_path, 'train.csv'), header=0)
#     # test = pd.read_csv(os.path.join(data_path, 'test.csv'), header=0)
#     return train

from sklearn import datasets, svm, metrics

#Load the digits dataset
digits = datasets.load_digits()
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])
# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))