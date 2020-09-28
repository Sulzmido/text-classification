"""
    To run this file, tesseract need to be installed.
    Visit. https://github.com/tesseract-ocr/tesseract/wiki#windows

    change pytesseract.pytesseract.tesseract_cmd to the directory tesseract is installed.

    Classify images via console [ tesseract +  Linear SVC ]
"""

from PIL import Image
import pytesseract
import pkg_resources
import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/model.joblib'))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

print('Kindly ensure the image you want to classify is in the images directory')

while True:
    user_input = input('Please specify the file name of the image you want to classify with extension e.g Noise.png '
                       '\n : ')

    filename = '../Images/' + user_input

    try:
        image = Image.open(filename)
    except FileNotFoundError:
        print('The selected file does not exist in the Images directory.')
        continue

    text = pytesseract.image_to_string(image)

    print('Extracted Text :', text)

    result = model.predict(vectorizer.transform([text]))

    if result[0] == 0:
        result = 'Not Profane'
    else:
        result = 'Profane'

    print('The image is :', result)

