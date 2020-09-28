"""
 This implementation is currently faulty.
"""

import pkg_resources
import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'Naive-Bayes Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'Naive-Bayes Approach/model.joblib'))

while True:
    user_input = input('Enter the word that needs to be classified : ')
    # print('1 - User Input:', user_input)
    inp = [user_input]
    # print('2 - User Input In Array:', inp)
    transformed = vectorizer.transform(inp)
    # print('3 - Vectorized Input:', transformed)
    transformed = transformed.toarray()
    # print('4 - Vector as Array:', transformed)

    result = model.predict(transformed)
    # print('5 - Prediction Result:', result)
    if result[0] == 0:
        print('not profane')
    else:
        print('profane')
