import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the ISOLET dataset
splice = fetch_openml(name='splice', version=1)

# Split the dataset into features and labels
X = splice.data
y = splice.target

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print unique values in y
print(np.unique(y))

import numpy as np

# Print unique values in each feature column
for i in range(X.shape[1]):
    unique_values = np.unique(X[:, i])
    print(f"Unique values in feature {i}: {unique_values}")