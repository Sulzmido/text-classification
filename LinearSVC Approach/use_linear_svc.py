import pkg_resources
import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/model.joblib'))

while True:
    user_input = input('Enter the word that needs to be classified : ')
    result = model.predict(vectorizer.transform([user_input]))
    if result[0] == 0:
        print('not profane')
    else:
        print('profane')
