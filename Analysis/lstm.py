import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from process_real import *

SEQUENCE_LENGTH = 20

# Step 1: Load the Normalized Data
data, labels = get_windows_scaled(SEQUENCE_LENGTH, overlapping=False, interpolate=True)
data = np.swapaxes(data, 1, 2)
print('data loaded')
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42, stratify=y)
print('data split')
# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Step 2: Create PyTorch Dataset and DataLoader
class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SensorDataset(X_train, y_train)
test_dataset = SensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print('datasets initialized')
# Step 5: Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        out = self.sigmoid(out)
        return out

input_size = X_train.shape[2]
hidden_size = 50
output_size = 1
num_layers = 2

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
print(model)

# Step 6: Define Loss Function and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 7: Train the Model
print('training starting')
epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output.squeeze(1), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")

# Step 8: Evaluate the Model
model.eval()
test_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch)
        loss = criterion(output.squeeze(1), y_batch)
        test_loss += loss.item()
test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Step 9: Visualize Predictions
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        output = model(X_batch).squeeze(1)
        predictions.append(output.numpy())
        actuals.append(y_batch.numpy())

plt.plot(actuals, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.show()
