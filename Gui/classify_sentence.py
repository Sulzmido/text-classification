"""
Classify sentences using trained LinearSVC
"""

import tkinter as tk
import pkg_resources
import joblib

vectorizer = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/vectorizer.joblib'))
model = joblib.load(pkg_resources.resource_filename('Project', 'LinearSVC Approach/model.joblib'))

text_color = 'black'


def classify_text():
    """Classify a text as profane or not-profane
    """
    word = ent_word.get("1.0", tk.END)
    result = model.predict(vectorizer.transform([word]))
    if result[0] == 0:
        lbl_classification["text"] = 'Result : Not profane.'
    else:
        lbl_classification["text"] = 'Result : Profane.'


window = tk.Tk()
window.title("Word Classifier")


def handle_keypress(event):
    """Reset classification."""
    # print(event.char)
    lbl_classification["text"] = ''


# Bind keypress event to handle_keypress()
window.bind("<Key>", handle_keypress)

frm_entry = tk.Frame(master=window)
lbl_instruction = tk.Label(master=frm_entry, text="Enter the sentence to classify :")
lbl_instruction.config(font=("Courier", 20))
ent_word = tk.Text(master=frm_entry, width=100, height=5)
lbl_instruction.grid(row=0, column=0, sticky="we")
ent_word.grid(row=1, column=0, sticky="we")
btn_convert = tk.Button(
    master=window,
    text="Classify    \N{RIGHTWARDS BLACK ARROW}",
    command=classify_text
)
lbl_classification = tk.Label(master=window, text="", fg=text_color)
lbl_classification.config(font=("Courier", 40))
frm_entry.grid(row=0, column=0, padx=10)
btn_convert.grid(row=1, column=0, padx=10, pady=10, sticky="w")
lbl_classification.grid(row=1, column=0, padx=10, sticky="e")

window.mainloop()
