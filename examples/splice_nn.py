import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from hdcpy_auxiliary import get_dataset, replace_features

import numpy as np

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

# Convert the numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets and dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # First fully connected layer
        self.relu = nn.ReLU()                  # ReLU activation
        self.fc2 = nn.Linear(128, num_classes) # Second fully connected layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

input_size = X_train.shape[1]  # Number of features in the dataset
num_classes = len(set(y_train))  # Number of unique classes

model = SimpleNN(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total:.2f}%')
