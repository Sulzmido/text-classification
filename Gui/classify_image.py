"""
Classify images using tesseract + trained LinearSVC
"""

from tkinter import Label, Tk, Button
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import pkg_resources
import joblib
import pytesseract

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/model.joblib'))

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\\tesseract.exe'

root = Tk()

values = {"text": "", "label": Label(root, cnf={'height': 600, 'width': 800})}
lbl_classification = Label(master=root, text="", fg='black')


def classify_text():
    """Classify a text as profane or not-profane
    """
    image_text_value = values["text"]
    result = model.predict(vectorizer.transform([image_text_value]))
    if result[0] == 0:
        lbl_classification["text"] = 'Result : Not profane.'
    else:
        lbl_classification["text"] = 'Result : Profane.'


def pick_image():
    path = askopenfilename(filetypes=[("Image Files", '.jpg;*.jpeg;*.png')])

    if not path:
        return

    im = Image.open(path)

    values["text"] = pytesseract.image_to_string(im)
    lbl_classification["text"] = ''

    tk_image = ImageTk.PhotoImage(im)

    label = values["label"]

    label.config(image=tk_image)
    label.image = tk_image
    label.pack()


pick_image()

btn_convert = Button(
    master=root,
    text="Classify    \N{RIGHTWARDS BLACK ARROW}",
    command=classify_text
)

btn_choose = Button(
    master=root,
    text="Choose Another",
    command=pick_image
)

btn_convert.pack()
btn_choose.pack()

lbl_classification.config(font=("Courier", 40))
lbl_classification.pack()

root.mainloop()
