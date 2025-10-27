from tkinter import *
from tkinter import filedialog, messagebox
import pickle
import re
import pandas as pd
import parselmouth
import csv
from sklearn.preprocessing import StandardScaler

# --- Setup main window ---
window = Tk()
window.geometry('300x300')
window.title("Parkinson's Detection")

# --- Global variable for filename ---
filename = None

# --- Browse File Function ---
def browse_file():
    global filename
    filename = filedialog.askopenfilename(
        title="Select Voice Sample",
        filetypes=[("WAV files", "*.wav")]
    )
    if filename:
        messagebox.showinfo("File Selected", f"Loaded: {filename}")
    else:
        messagebox.showwarning("No File", "Please select a valid WAV file.")

# --- Help Function ---
def help_me():
    messagebox.showinfo("Help", "Step 1: Click 'File → Open' to select a WAV file.\nStep 2: Click 'Detect' to analyze Parkinson's disease.")

# --- Detect Function ---
def detect():
    global filename
    if not filename:
        messagebox.showerror("Error", "Please select a WAV file first!")
        return

    try:
        # Process audio
        sound = parselmouth.Sound(filename)
        pitch = sound.to_pitch()
        pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
        voice_report_str = parselmouth.praat.call(
            [sound, pitch, pulses],
            "Voice report", 0.0, 0.0, 75, 600, 1.3, 1.6, 0.03, 0.45
        )

        # Extract numeric values
        s = re.findall(r'-?\d+\.?\d*', voice_report_str)
        df = pd.DataFrame(s)
        df.to_csv("file_path.csv")

        # Replace row in test.csv
        row = ['1', s[21], s[22]+'E'+s[23], s[24], s[26], s[27], s[28], s[29],
               s[31], s[33], s[35], s[36], s[37], s[38], s[39], s[3], s[4],
               s[5], s[6], s[7], s[8], s[9], s[10]+'E'+s[11], s[12]+'E'+s[13]]

        with open('test.csv', 'r') as readFile:
            reader = csv.reader(readFile)
            lines = list(reader)
            lines[1] = row

        with open('test.csv', 'w', newline='') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(lines)

        # Load trained model
        svm_pkl_filename = 'svmclassifier.pkl'
        with open(svm_pkl_filename, 'rb') as file:
            svm_model = pickle.load(file)

        # Predict
        data = pd.read_csv('test.csv')
        X_test1 = data.iloc[:, [5, 23, 22, 13, 1, 7, 2, 4, 12, 3]].values

        y_pred = svm_model.predict(X_test1)
        print("Prediction:", y_pred)

        # Display result
        if y_pred[0] == 1:
            messagebox.showinfo("Result", "You have been diagnosed with Parkinson's Disease.")
        else:
            messagebox.showinfo("Result", "You are a healthy person.")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

# --- Menu Bar ---
menubar = Menu(window)
submenu = Menu(menubar, tearoff=0)
window.config(menu=menubar)

menubar.add_cascade(label="File", menu=submenu)
submenu.add_command(label="Open", command=browse_file)
submenu.add_command(label="Exit", command=window.destroy)

submenu2 = Menu(menubar, tearoff=0)
menubar.add_cascade(label="About Us", menu=submenu2)
submenu2.add_command(label="Help", command=help_me)

# --- Detect Button ---
detectbutton = Button(window, text="Detect", command=detect)
detectbutton.pack(pady=40)

window.mainloop()
