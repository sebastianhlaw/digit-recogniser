# Author:   Sebastian Law
# Date:     30-Jun-2016
# Revised:  08-Jul-2016

import loader
from sklearn import svm, model_selection
from sklearn.ensemble import RandomForestClassifier


X, y = loader.load_sklearn()
# X, y = loader.load_kaggle_public()
# Split full data into 75% public (train/validate) and 25% test
X_data, X_test, y_data, y_test = model_selection.train_test_split(
    X, y, test_size=0.25, random_state=0)

# Split the public data into train and test - 50% train, 25% validate
X_train, X_validate, y_train, y_validate = model_selection.train_test_split(
    X_data, y_data, test_size=1.0/3.0, random_state=0)

# get an SVM benchmark
# classifier = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
classifier = RandomForestClassifier(n_estimators=1000).fit(X_train, y_train)
val_score = classifier.score(X_validate, y_validate)
print("val_score", val_score)

# # investigate further
# y_predict = classifier.predict(X_validate)
# print(metrics.classification_report(y_validate, y_predict))
# print(metrics.confusion_matrix(y_validate, y_predict))
# misclassified = []
# for i, z in enumerate(y_test):
#     if y_test[i] != y_predict[i]: misclassified.append(i)


# final checks
test_score = classifier.score(X_test, y_test)
print("test_score", test_score)







