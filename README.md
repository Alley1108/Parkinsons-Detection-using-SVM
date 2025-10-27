# Parkinsons-Detection-using-SVM
Machine Learning–based Parkinson’s Disease Detection system using SVM classifier trained on biomedical voice data from the UCI dataset. Extracts vocal features via Parselmouth (Praat) and predicts Parkinson’s through a Tkinter GUI for real-time voice-based diagnosis.


Parkinson’s Disease Detection – File Descriptions
1️⃣ 1.py – Model Training Script

This file is responsible for training the Support Vector Machine (SVM) classifier.

Imports and reads the dataset (final1.csv).

Selects key biomedical voice parameters (like jitter, shimmer, HNR, etc.) as input features.

Splits the dataset into training (80%) and testing (20%) sets.

Standardizes data using StandardScaler to normalize feature ranges.

Trains a linear SVM model to classify between healthy and Parkinson’s-affected voices.

Evaluates performance using cross-validation and confusion matrix.

Saves the trained model as a pickle file (svmclassifier.pkl) for later use.

2️⃣ 2.py – Feature Extraction and Prediction Script

This script extracts features from a voice recording (Phonation_2.wav) and predicts whether it indicates Parkinson’s disease.

Loads the .wav audio using the Parselmouth (Praat) library for speech analysis.

Extracts acoustic features such as pitch, jitter, shimmer, and noise ratios from the voice.

Writes these extracted values into test.csv for consistency with the training dataset.

Loads the pre-trained SVM model (svmclassifier.pkl).

Predicts the disease status based on the extracted voice features (output: 1 = Parkinson’s, 0 = Healthy).

This file essentially simulates real-time detection without a GUI.

3️⃣ final1.csv – Training Dataset

Contains around 1,000 samples of biomedical voice measurements.

Each row represents one individual’s voice data with 24 numerical parameters derived from speech recordings.

Includes a “status” column as the target label:

1 → Parkinson’s Disease

0 → Healthy Control

Features include: MDVP:Fo(Hz), MDVP:Jitter(%), MDVP:Shimmer, NHR, HNR, RPDE, DFA, spread1, spread2, PPE, etc.
This dataset was adapted from the UCI Machine Learning Repository’s Parkinson’s Disease dataset.

4️⃣ gui.py – Graphical User Interface (Tkinter)

Provides a user-friendly desktop interface for detection.

Built using Tkinter, Python’s standard GUI library.

Allows users to browse and select a .wav voice file.

On clicking “Detect”, the program:

Extracts voice features using Parselmouth.

Updates test.csv with these new features.

Loads the pre-trained SVM model (svmclassifier.pkl).

Predicts the disease outcome.

Displays a popup message — “Healthy Person” or “Diagnosed with Parkinson’s Disease”.
This makes the project interactive and easy to use, without requiring command-line execution.

5️⃣ Phonation_2.wav – Sample Voice File

A test voice sample used for feature extraction and prediction.

Represents the kind of input the model analyzes (sustained vowel phonation, typically “ahh”).

Used by both 2.py and gui.py for demonstration and testing.

6️⃣ svmclassifier.pkl – Trained Model File

The serialized SVM model created by 1.py.

Stores the trained model parameters, decision boundaries, and learned patterns from the dataset.

Allows instant prediction without re-training.

Loaded during testing (2.py) or GUI detection (gui.py).

7️⃣ test.csv – Testing Data File

Used to store extracted voice features from the user’s input audio.

Has the same column structure as the training dataset.

Gets automatically updated each time a new .wav file is processed.

Serves as the input file for the prediction process in both 2.py and gui.py.

✅ Overall Workflow Summary

Train Model → Run 1.py using final1.csv.
⮕ Generates svmclassifier.pkl.

Predict Manually → Run 2.py with Phonation_2.wav.
⮕ Updates test.csv and prints the result.

Predict via GUI → Run gui.py for an easy voice upload and instant detection interface.
