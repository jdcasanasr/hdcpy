import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from hdcpy_auxiliary import get_dataset, replace_features

dataset_name    = 'splice'
save_directory  = '../data'
test_proportion = 0.2

features_dictionary = {'A': 0, 'G': 1, 'T': 2, 'C': 3, 'D': 4, 'N': 5, 'S': 6, 'R': 7}
labels_dictionary   = {'N': 0, 'EI': 1, 'IE': 2}

X_train, X_test, y_train, y_test = get_dataset(dataset_name, save_directory, test_proportion)

X_train     = replace_features(X_train, features_dictionary).astype(np.int_)
X_test      = replace_features(X_test, features_dictionary).astype(np.int_)
y_train     = replace_features(y_train, labels_dictionary).astype(np.int_)
y_test      = replace_features(y_test, labels_dictionary).astype(np.int_)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)