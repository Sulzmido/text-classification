# Accuracy tests
# Precision tests
# Recall tests
# F1 Scores

import pandas as pd
import pkg_resources
import joblib
from sklearn.metrics import accuracy_score, classification_report

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/model.joblib'))

# Read in data
data = pd.read_csv('../Data/clean_data_test.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

y_true = y

transformed = vectorizer.transform(texts)
# print(transformed)

y_pred = model.predict(transformed)
# print(y_pred)

print('Linear SVC Classification Report')

target_names = ['Not-Profane', 'Profane']
print(classification_report(y_true, y_pred, target_names=target_names))

accuracy = accuracy_score(y_true, y_pred)
print('Linear SVC Accuracy ', accuracy)
