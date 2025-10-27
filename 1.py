import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

# Load dataset
dataset = pd.read_csv('final1.csv')

# Print column names to confirm label column
print(dataset.columns)

# Select features and labels
X = dataset.iloc[:, [5,23,22,13,1,7,2,4,8,3]].values
y = dataset.iloc[:, 24].values
y = y.astype(int)   # <--- FIX HERE

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train SVM
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Evaluate
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy:", accuracies.mean())

y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Save model
pickle.dump(classifier, open('svmclassifier.pkl', 'wb'))
print("Model saved as svmclassifier.pkl ✅")
