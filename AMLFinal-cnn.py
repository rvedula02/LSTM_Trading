import yfinance as yf
import talib as ta
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import copy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Fetch and Prepare the Data
ticker = 'AAPL'
market_ticker = '^GSPC'  # S&P 500
start_date = '2016-01-01'
end_date = '2023-01-01'

# Download both stock and market data
data = yf.download(ticker, start=start_date, end=end_date)
market_data = yf.download(market_ticker, start=start_date, end=end_date)

# Function to calculate technical indicators
def calculate_technical_indicators(df):
    close = df['Close'].values.astype(float).ravel()
    high = df['High'].values.astype(float).ravel()
    low = df['Low'].values.astype(float).ravel()
    volume = df['Volume'].values.astype(float).ravel()
    
    df['OBV'] = ta.OBV(close, volume)
    df['SMA_20'] = ta.SMA(close, timeperiod=20)
    macd, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    df['RSI'] = ta.RSI(close, timeperiod=14)
    df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
    
    return df

# Calculate indicators for both datasets
data = calculate_technical_indicators(data)
market_data = calculate_technical_indicators(market_data)

# Prepare Features and Targets
scaler = StandardScaler()
features = pd.concat([
    data[['Close', 'SMA_20', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI', 'ATR', 'OBV']],
    market_data[['Close', 'SMA_20', 'MACD', 'MACD_signal', 'MACD_hist', 'RSI', 'ATR', 'OBV']].add_prefix('SPX_')
], axis=1)

# Handle any NaN values
features = features.fillna(method='ffill').fillna(method='bfill')

scaled_features = scaler.fit_transform(features)
price_data = data['Close'].values
price_scaler = StandardScaler()
scaled_price_data = price_scaler.fit_transform(price_data.reshape(-1, 1))

# Function to create sequences
def create_sequences(data, price, sequence_length=20):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(price[i + sequence_length])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X_features, y_features = create_sequences(scaled_features, scaled_price_data)

# Split the data into training, validation, and test sets
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_features, y_features, test_size=0.1, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, shuffle=False)

# Create DataLoaders
train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_data = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Define CNN Model
class CNNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout_rate):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_channels, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change to (batch, features, seq_len)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.mean(x, dim=2)  # Global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Define the hyperparameter search space
search_space = {
    'num_channels': [16, 32, 64],
    'kernel_size': [2, 3, 4],
    'dropout_rate': [0.3, 0.4, 0.5],
    'optimizer': ['Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD'],
    'learning_rate': [0.01, 0.001, 0.0001]
}

# ARO Hyperparameter Optimization
class AROOptimizer:
    def __init__(self, num_rabbits, max_iter, search_space):
        self.num_rabbits = num_rabbits
        self.max_iter = max_iter
        self.search_space = search_space

    def optimize(self, train_data, val_data, input_size):
        best_score = float("inf")
        best_params = None
        rabbits = [self.random_params() for _ in range(self.num_rabbits)]
        scores = [float("inf")] * self.num_rabbits

        for iter in range(self.max_iter):
            print(f"Iteration {iter+1}/{self.max_iter}")
            for i, rabbit in enumerate(rabbits):
                print(f"  Evaluating rabbit {i+1}/{len(rabbits)}")
                model = CNNModel(input_size, rabbit['num_channels'], rabbit['kernel_size'], rabbit['dropout_rate'])
                optimizer_class = getattr(torch.optim, rabbit['optimizer'])
                optimizer = optimizer_class(model.parameters(), lr=rabbit['learning_rate'])
                loss = self.train_and_evaluate(model, train_data, val_data, optimizer)
                print(f"    Validation Loss: {loss:.6f}")
                if loss < scores[i]:
                    scores[i] = loss
                if loss < best_score:
                    best_score = loss
                    best_params = rabbit

        return best_params

    def random_params(self):
        return {
            'num_channels': random.choice(self.search_space['num_channels']),
            'kernel_size': random.choice(self.search_space['kernel_size']),
            'dropout_rate': random.choice(self.search_space['dropout_rate']),
            'optimizer': random.choice(self.search_space['optimizer']),
            'learning_rate': random.choice(self.search_space['learning_rate'])
        }

    def train_and_evaluate(self, model, train_data, val_data, optimizer):
        criterion = nn.MSELoss()
        model.train()
        epochs = 10  # Adjust as needed
        for epoch in range(epochs):
            for X_batch, y_batch in train_data:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch.squeeze())
                loss.backward()
                optimizer.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_data:
                output = model(X)
                val_loss += criterion(output.squeeze(), y.squeeze()).item()
        return val_loss / len(val_data)

# Run ARO Optimization
aro = AROOptimizer(num_rabbits=10, max_iter=5, search_space=search_space)
best_params = aro.optimize(train_data, val_data, input_size=X_features.shape[2])

print("Best Parameters found by ARO:", best_params)

X_combined = torch.cat((X_train, X_val), dim=0)
y_combined = torch.cat((y_train, y_val), dim=0)
combined_data = DataLoader(TensorDataset(X_combined, y_combined), batch_size=32, shuffle=True)

# Build and Train Final Model with Best Parameters
best_model = CNNModel(input_size=X_features.shape[2], num_channels=best_params['num_channels'], 
                      kernel_size=best_params['kernel_size'], dropout_rate=best_params['dropout_rate'])
optimizer_class = getattr(torch.optim, best_params['optimizer'])
optimizer = optimizer_class(best_model.parameters(), lr=best_params['learning_rate'])
criterion = nn.MSELoss()
epochs = 50  # Adjust as needed
for epoch in range(epochs):
    best_model.train()
    for X_batch, y_batch in combined_data:
        optimizer.zero_grad()
        output = best_model(X_batch)
        loss = criterion(output.squeeze(), y_batch.squeeze())
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}")

def make_predictions(model, test_data, use_predictions=False, window_size=5):
    model.eval()
    predictions = []
    actuals = []

    # Initialize with test data
    current_sequence = X_test[0].unsqueeze(0)  # Start with first test sequence
    sequence_length = current_sequence.shape[1]
    feature_size = current_sequence.shape[2]
    
    with torch.no_grad():
        for i in range(len(test_data.dataset)):
            # Get actual values
            actual = y_test[i].numpy()
            actuals.append(actual)

            # Make prediction
            output = model(current_sequence)
            pred = output.numpy()
            predictions.append(pred)

            # Update sequence for next prediction
            if use_predictions:
                # Prepare the new features using the prediction
                last_sequence = current_sequence[0, -1, :].numpy()  # Get last timestep's features
                last_sequence[0] = pred  # Update only the price, keep other features
                new_features = last_sequence.reshape(1, 1, feature_size)
                
                # Update the sequence
                current_sequence = torch.cat([
                    current_sequence[:, 1:, :],
                    torch.FloatTensor(new_features)
                ], dim=1)
            else:
                # Use actual values
                if i < len(test_data.dataset) - 1:
                    current_sequence = X_test[i + 1].unsqueeze(0)  # Use the actual next sequence

    predictions = np.array(predictions).reshape(-1, 1)
    actuals = np.array(actuals).reshape(-1, 1)

    # Inverse transform predictions and actuals
    predictions = price_scaler.inverse_transform(predictions)
    actuals = price_scaler.inverse_transform(actuals)

    return predictions, actuals

def evaluate_predictions(pred_prices, actual_prices, mode=""):
    mae = mean_absolute_error(actual_prices, pred_prices)
    mse = mean_squared_error(actual_prices, pred_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, pred_prices)
    
    # Calculate directional accuracy
    direction_metrics = calculate_direction_accuracy(actual_prices, pred_prices)

    print(f"\n{mode} Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Directional Accuracy: {direction_metrics['accuracy']:.2f}%")
    print("\nDirectional Movement Analysis:")
    print(f"Total Moves: {direction_metrics['total_moves']}")
    print(f"Correct Directions: {direction_metrics['correct_directions']}")
    print(f"True Ups: {direction_metrics['true_ups']}")
    print(f"True Downs: {direction_metrics['true_downs']}")
    print(f"False Ups: {direction_metrics['false_ups']}")
    print(f"False Downs: {direction_metrics['false_downs']}")
    
    return mae, mse, rmse, r2, direction_metrics

def calculate_direction_accuracy(actual_prices, pred_prices):
    """
    Calculate the accuracy of predicted price movement direction.
    Returns both accuracy and detailed movement analysis.
    """
    actual_moves = np.sign(actual_prices[1:] - actual_prices[:-1])
    pred_moves = np.sign(pred_prices[1:] - pred_prices[:-1])
    
    # Calculate accuracy
    correct_directions = np.sum(actual_moves == pred_moves)
    total_moves = len(actual_moves)
    accuracy = (correct_directions / total_moves) * 100

    # Detailed analysis
    true_ups = np.sum((actual_moves == 1) & (pred_moves == 1))
    true_downs = np.sum((actual_moves == -1) & (pred_moves == -1))
    false_ups = np.sum((actual_moves == -1) & (pred_moves == 1))
    false_downs = np.sum((actual_moves == 1) & (pred_moves == -1))
    
    return {
        'accuracy': accuracy,
        'total_moves': total_moves,
        'correct_directions': correct_directions,
        'true_ups': true_ups,
        'true_downs': true_downs,
        'false_ups': false_ups,
        'false_downs': false_downs
    }

def plot_predictions(pred_prices_actual, pred_prices_pred, actual_prices):
    plt.figure(figsize=(15, 7))
    plt.plot(actual_prices, label='Actual Prices', color='black')
    plt.plot(pred_prices_actual, label='Predictions (Using Actual Values)', color='blue')
    plt.plot(pred_prices_pred, label='Predictions (Using Prior Predictions)', color='red')
    plt.title('Stock Price Predictions: Actual vs Predicted Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run both prediction methods
predictions_actual, actuals = make_predictions(best_model, test_data, use_predictions=False)
predictions_pred, _ = make_predictions(best_model, test_data, use_predictions=True)

# Evaluate both methods
print("\nEvaluation Results:")
print("=" * 50)
metrics_actual = evaluate_predictions(predictions_actual, actuals, "Using Actual Values")
print("=" * 50)
metrics_pred = evaluate_predictions(predictions_pred, actuals, "Using Prior Predictions")
print("=" * 50)

# Plot results
plot_predictions(predictions_actual, predictions_pred, actuals)
