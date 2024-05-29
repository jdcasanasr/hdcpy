import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from hdcpy_auxiliary import get_dataset

# Load the dataset
dataset_name    = 'cardiotocography'
save_directory  = '../data'
test_proportion = 0.2

X_train, X_test, y_train, y_test = get_dataset(dataset_name, save_directory, test_proportion)

# Standardize the features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
y_train = list(map(int, y_train))
y_test  = list(map(int, y_test))

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype = torch.float32)
X_test  = torch.tensor(X_test,  dtype = torch.float32)
y_train = torch.tensor(y_train, dtype = torch.long)
y_test  = torch.tensor(y_test,  dtype = torch.long)

# Define the neural network
class CardioNet(nn.Module):
    def __init__(self):
        super(CardioNet, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)  # Assuming 3 classes for the target variable
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, define the loss function and the optimizer
model = CardioNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test, predicted)
    print(f'Accuracy: {accuracy:.4f}')