# Author:   Sebastian Law
# Date:     16-Nov-2016
# Revised:  16-Dec-2016

from sklearn import ensemble, model_selection
from sklearn.decomposition import PCA

import loader
import factor_analysis
import results


# Load the digits dataset - it is 60% public, 40% private/test
print("Loading data...")
X_data, y_data = loader.load_kaggle_public()
X_test = loader.load_kaggle_private()

# Split the public data into train and validate, this is a benchmark
X_train, X_validate, y_train, y_validate = model_selection.train_test_split(
    X_data, y_data, train_size=0.25, test_size=0.05, random_state=0)

# run PCA to reduce dimensionality, capture 95% of variance
print("PCA...")
dimension = factor_analysis.pca_components(X_train, percentile=0.95, plot=True)
print("Dimension for PCA is:", dimension)
pca = PCA(n_components=dimension, whiten=True)
XX_train = pca.fit(X_train).transform(X_train)
XX_validate = pca.transform(X_validate)
XX_test = pca.transform(X_test)


# run the classifier
print("Running classifier...")
classifier = ensemble.RandomForestClassifier(n_estimators=1000).fit(XX_train, y_train)
score = classifier.score(XX_train, y_train)
print("Training score:", score)
score = classifier.score(XX_validate, y_validate)
print("Validation score:", score)
y_predict = classifier.predict(XX_test)
results.dump(y_predict, "benchmark_randomforest")

print("Finished.")
