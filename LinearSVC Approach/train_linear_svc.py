"""
LinearSVC can handle more training data in comparison to Naive-Bayes.
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import joblib

# Read in data
data = pd.read_csv('../Data/clean_data_train.csv')  # train with clean_data.csv for better accuracy.
texts = data['text'].astype(str)
y = data['is_offensive']

# print(texts)

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(texts)

# print(X)

# Train the model using LinearSVC
model = LinearSVC(class_weight="balanced", dual=False, tol=1e-2, max_iter=1e5)
cclf = CalibratedClassifierCV(base_estimator=model)
cclf.fit(X, y)

# Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(cclf, 'model.joblib')
