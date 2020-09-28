import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
import joblib

# Read in data
data = pd.read_csv('../Data/clean_data_train.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# print(texts)

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(texts)

# print(X)
X = X.toarray()

# Train the model using Naive-Bayes
model = GaussianNB()
model.fit(X, y)

# Save the model
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(model, 'model.joblib')
