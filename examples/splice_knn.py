import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

# Define the KNN classifier
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Train the KNN classifier
knn.fit(X_train, y_train)

# Predict labels for the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:0.2f}')