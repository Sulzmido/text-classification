# Accuracy tests
# Precision tests
# Recall tests
# F1 Scores

import pandas as pd
import pkg_resources
import joblib
from sklearn.metrics import accuracy_score, classification_report

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'Naive-Bayes Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'Naive-Bayes Approach/model.joblib'))

# Read in data
data = pd.read_csv('../Data/clean_data_test.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

y_true = y

# print(texts)
transformed = vectorizer.transform(texts)
# print(transformed)
transformed = transformed.toarray()
# print(transformed)

y_pred = model.predict(transformed)
# print(y_pred)

print('Naive-Bayes Classification Report')

target_names = ['Not-Profane', 'Profane']
print(classification_report(y_true, y_pred, target_names=target_names))

accuracy = accuracy_score(y_true, y_pred)
print('Naive-Bayes Accuracy ', accuracy)
