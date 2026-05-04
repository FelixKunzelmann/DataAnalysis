import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Fetch Bitcoin data
ticker = 'BTC-USD'
data = yf.download(ticker, start='2019-01-01', end='2024-01-01')
data = data[['Close']]

# Preprocess data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Create training data


def create_dataset(dataset, time_step=60):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)


time_step = 60
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split into train and test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


model = LSTMModel()

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(
            f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Predict on test data
model.eval()
with torch.no_grad():
    predictions_tensor = model(X_test_tensor)
    predictions = predictions_tensor.numpy()

predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate accuracy
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'RMSE: {rmse}')


# Predict future days (next 30 days)
model.eval()
last_60_days = torch.FloatTensor(scaled_data[-60:]).view(1, -1, 1)
future_predictions = []

with torch.no_grad():
    for _ in range(30):
        pred = model(last_60_days)
        future_predictions.append(pred.item())
        # Update last_60_days by removing first element and adding prediction
        last_60_days = torch.cat(
            [last_60_days[:, 1:, :], pred.view(1, 1, 1)], dim=1)

future_predictions = scaler.inverse_transform(
    np.array(future_predictions).reshape(-1, 1))
print('Predicted prices for next 30 days:')
for i, price in enumerate(future_predictions):
    print(f'Day {i+1}: {price[0]}')

# Plot predictions vs actual
plt.figure(figsize=(14, 5))
plt.plot(y_test_actual, color='blue', label='Actual Bitcoin Price')
plt.plot(predictions, color='red', label='Predicted Bitcoin Price')
plt.title('Bitcoin Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Second plot: Last 60 days actual + last 30 days predicted + next 30 days predicted
plt.figure(figsize=(14, 5))
last_60_actual = data[-60:].values.flatten()
future_dates = pd.date_range(
    start=data.index[-1] + pd.Timedelta(days=1), periods=30)

# Get dates and predictions for the last 30 days of test data
test_start_idx = train_size + time_step
test_dates = data.index[test_start_idx: test_start_idx + len(predictions)]
last_30_test_dates = test_dates[-30:]
last_30_predictions = predictions[-30:]

plt.plot(data.index[-60:], last_60_actual,
         color='blue', label='Last 60 Days Actual')
plt.plot(last_30_test_dates, last_30_predictions.flatten(),
         color='green', label='Predicted (Last 30 Test Days)')
plt.plot(future_dates, future_predictions.flatten(),
         color='red', label='Next 30 Days Predicted')
plt.title('Bitcoin Price: Last 60 Days Actual + Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
