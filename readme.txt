# Detection of Parkinson's Disease Using Machine Learning.
## Parkison's - A disorder of the central nervous system that affects movement, often including tremors


parkinson’s disease can be detected using speech. The phonation is the most affected part of the speech i.e. The sound we make when we pronounce the vowels. We have used the database of the speech samples containing the phonation from the affected and healthy people. Various database of the speech sample is available from JASA (Journal of Acoustic Society of America), UCI. 
Speech signals or the voice samples have been taken from the standard UCI voice dataset which consists of voice samples of people. The samples of healthy people are also collected for the comparative study. The Test data belongs to 56 subjects. During the collection of this dataset, 56 people are asked to say only the sustained vowels 'a' and 'o' three times respectively. Total of 336 recordings are obtained from the repository. 
In the training phase the pre-processing of these signals is done for feature extraction by PRAAT software. The features extracted are jitter, shimmer, NHR, HNR, mean and median pitch, number of pulses and periods, minimum and maximum period, SD, SD of period, number and degree of voice breaks.
All these features differ from patient to patient depending upon the fact how much Parkinson’s disease has progressed. 
After extracting all the features we will do dimensionality reduction of the features using particle swarm optimization(PSO),
In this optimization method it works like swarm particle and reduce the features selection to a minimum, optimization involves in achieving better result in less computation, after selection of the features, The features are used to train the SVM classifier and the model is  trained.


🧠 Parkinson’s Disease Detection using Machine Learning

Tools & Technologies: Python, scikit-learn, Tkinter (GUI), Parselmouth, Pandas, NumPy, CSV, SVM

Project Overview:
Developed a machine learning–based system to detect Parkinson’s disease from voice recordings using biomedical signal analysis. The model predicts whether a person is healthy or Parkinsonian by analyzing vocal parameters extracted from speech samples.

Key Components:

Data Source: Imported dataset containing approximately 1,000 voice samples (rows) and 24 biomedical voice parameters such as MDVP:Fo(Hz), MDVP:Jitter(%), Shimmer, HNR, RPDE, DFA, etc.

Feature Extraction: Extracted acoustic features (pitch, jitter, shimmer, harmonic-to-noise ratio) from .wav files using the Parselmouth (Praat) library for phonetic signal processing.

Model Used: Trained a Support Vector Machine (SVM) classifier on the dataset to distinguish between healthy and Parkinson’s-affected individuals.

Model Storage: Serialized the trained model using pickle (svmclassifier.pkl) for later use in real-time predictions.

Testing Data: Used test.csv for validation, ensuring consistent performance across unseen samples.

GUI Implementation: Built a Tkinter-based desktop interface allowing users to upload a .wav voice sample. The system automatically extracts vocal features, applies the trained SVM model, and displays a result message (“Healthy” / “Parkinson’s detected”).

Performance Metrics: Model achieved high classification accuracy (typically 90%+ on validation data).

Objective: To demonstrate how machine learning and signal processing can be applied in healthcare diagnostics for early disease detection using non-invasive methods.

Skills Highlighted:

Supervised learning (SVM classification)

Audio signal feature extraction (Praat/Parselmouth)

Data preprocessing and analysis (Pandas, NumPy)

Model deployment using GUI (Tkinter)

Handling CSV datasets and model persistence with pickle

📄 Short Resume Version (for bullet point section)

Developed a Parkinson’s Disease Detection System using SVM classifier trained on a voice dataset (~1,000 samples, 24 features).

Implemented feature extraction from .wav files using Parselmouth (Praat) to derive acoustic parameters (jitter, shimmer, pitch).

Built a Tkinter-based GUI for real-time diagnosis by uploading a voice file and predicting results using a pre-trained model (svmclassifier.pkl).

Technologies used: Python, scikit-learn, Pandas, Tkinter, Parselmouth, pickle.












#DATASET
🧾 Dataset Source

Name: Parkinson’s Disease Data Set
Repository: UCI Machine Learning Repository
Link (for your reference, not to include in CV):
👉 https://archive.ics.uci.edu/ml/datasets/parkinsons

Creators:
Max A. Little, Patrick E. McSharry, Eric J. Hunter, Jennifer Spielman, and Lorraine O. Ramig.

📊 Dataset Details

Number of instances: 195 samples (in original UCI dataset)
→ Some GitHub projects expand it to ~1,000 entries by combining multiple .csv versions or augmentations.

Number of attributes: 24 biomedical voice parameters + 1 target label

Target column: status

1 → Parkinson’s disease present

0 → Healthy individual