# Author:   Sebastian Law
# Date:     22-Nov-2016
# Revised:  22-Nov-2016

import loader
from sklearn.ensemble import RandomForestClassifier
import results

# Load the digits dataset - 60% public, 40% private/test
print("Loading data...")
X_data, y_data = loader.load_kaggle_public()
X_test = loader.load_kaggle_private()

# run the classifier
print("Running classifier...")
classifier = RandomForestClassifier(n_estimators=1000).fit(X_data, y_data)
score = classifier.score(X_data, y_data)
print("Score:", score)

print("Generating results...")
y_predict = classifier.predict(X_test)
results.dump(y_predict, "benchmark_rf")

print("Finished.")