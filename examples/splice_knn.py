import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from hdcpy_auxiliary import get_dataset, substitute_characters, sub_2

dataset_name    = 'splice'
save_directory  = '../data'
test_proportion = 0.2

X_train, X_test, y_train, y_test = get_dataset(dataset_name, save_directory, test_proportion)
# Encode labels
#label_encoder = LabelEncoder()
#y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for index, feature_vector in enumerate(X_train):
    X_train[index] = substitute_characters(feature_vector)

for index, feature_vector in enumerate(X_test):
    X_test[index] = substitute_characters(feature_vector)

for index, label in enumerate(y_train):
    y_train[index] = sub_2(label)

for index, label in enumerate(y_test):
    y_test[index] = sub_2(label)

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