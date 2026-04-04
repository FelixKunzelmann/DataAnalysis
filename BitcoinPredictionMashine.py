import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

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
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=40, batch_size=32, verbose=1)

# Predict on test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate accuracy
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'RMSE: {rmse}')


# Predict future days (next 30 days)
last_60_days = scaled_data[-60:]
future_predictions = []
for _ in range(30):
    input_data = last_60_days.reshape(1, -1, 1)
    pred = model.predict(input_data)
    future_predictions.append(pred[0][0])
    last_60_days = np.append(last_60_days[1:], pred[0][0])

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
