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
    # Convert inputs to numpy arrays and ensure they're float type and 1D
    close = df['Close'].values.astype(float).ravel()
    high = df['High'].values.astype(float).ravel()
    low = df['Low'].values.astype(float).ravel()
    volume = df['Volume'].values.astype(float).ravel()
    
    # Print shapes for debugging
    print(f"Close shape: {close.shape}")
    print(f"Volume shape: {volume.shape}")
    
    # Ensure arrays are not empty
    if len(close) == 0 or len(volume) == 0:
        raise ValueError("Input arrays are empty")
    
    try:
        # Volume-based indicators
        df['OBV'] = ta.OBV(close, volume)
        
        # Trend indicators
        df['SMA_20'] = ta.SMA(close, timeperiod=20)
        macd, macd_signal, macd_hist = ta.MACD(close, 
                                              fastperiod=12, 
                                              slowperiod=26, 
                                              signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macd_signal
        df['MACD_hist'] = macd_hist
        
        # Momentum indicators
        df['RSI'] = ta.RSI(close, timeperiod=14)
        
        # Volatility indicators
        df['ATR'] = ta.ATR(high, low, close, timeperiod=14)
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        raise
    
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
# Ensure that we don't shuffle the data to maintain the temporal order
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_features, y_features, test_size=0.1, shuffle=False)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, shuffle=False)

# Create DataLoaders
train_data = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_data = DataLoader(TensorDataset(X_val, y_val), batch_size=32)
test_data = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

# Define LSTM Model with Flexible Architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, layer_params):
        super(LSTMModel, self).__init__()
        
        self.layers = nn.ModuleList()
        neurons_prev = input_size
        self.lstm_layers = []
        
        # Add L2 regularization through weight decay in optimizer
        self.weight_decay = 0.01  # L2 regularization strength
        
        for i, (layer_exist, neurons) in enumerate(layer_params['lstm_layers']):
            if layer_exist:
                lstm_layer = nn.LSTM(neurons_prev, neurons, batch_first=True)
                self.layers.append(lstm_layer)
                self.lstm_layers.append(lstm_layer)
                neurons_prev = neurons
                if layer_params['dropout_rate'] > 0:
                    self.layers.append(nn.Dropout(layer_params['dropout_rate']))
                # Add batch normalization with correct dimension
                self.layers.append(nn.BatchNorm1d(neurons))

        self.fc = nn.Linear(neurons_prev, 1)
        
        optimizer_class = getattr(torch.optim, layer_params['optimizer'])
        self.optimizer = optimizer_class(
            self.parameters(), 
            lr=layer_params['learning_rate'],
            weight_decay=self.weight_decay
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        for layer in self.layers:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            elif isinstance(layer, nn.BatchNorm1d):
                # Reshape for BatchNorm1d
                x = x.permute(0, 2, 1)  # Change to (batch, features, seq_len)
                x = layer(x)
                x = x.permute(0, 2, 1)  # Change back to (batch, seq_len, features)
            else:
                x = layer(x)
        
        # Use the final output
        x = self.fc(x[:, -1, :])  # Only take the last time step
        return x

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
                model = LSTMModel(input_size, rabbit)
                loss = self.train_and_evaluate(model, train_data, val_data)
                print(f"    Validation Loss: {loss:.6f}")
                if loss < scores[i]:
                    scores[i] = loss
                if loss < best_score:
                    best_score = loss
                    best_params = rabbit

        sorted_indices = np.argsort(scores[:len(rabbits)])  # Ensure scores is trimmed to rabbits length if necessary
        rabbits = [rabbits[i] for i in sorted_indices]
        scores = [scores[i] for i in sorted_indices]

        # Keep the top-performing rabbits and generate new ones
        num_keep = self.num_rabbits // 2
        best_rabbits = rabbits[:num_keep]
        rabbits = best_rabbits.copy()
        scores = [float("inf")] * len(rabbits)  # Reset scores to maintain alignment

        # Add new rabbits to maintain the original rabbit count
        while len(rabbits) < self.num_rabbits:
            parent = random.choice(best_rabbits)
            new_rabbit = self.perturb_params(parent)
            rabbits.append(new_rabbit)
            scores.append(float("inf"))
        return best_params

    def random_params(self):
        return {
            'lstm_layers': [(random.choice([0, 1]), random.choice(self.search_space['num_neurons'])) for _ in range(3)],
            'dropout_rate': random.choice(self.search_space['dropout_rate']),
            'optimizer': random.choice(self.search_space['optimizer']),
            'learning_rate': random.choice(self.search_space['learning_rate'])
        }

    def perturb_params(self, params):
        new_params = copy.deepcopy(params)
        # Perturb 'lstm_layers'
        new_layers = []
        for layer_exist, neurons in new_params['lstm_layers']:
            if layer_exist:
                idx = self.search_space['num_neurons'].index(neurons)
                if random.random() < 0.5:
                    if random.random() < 0.5 and idx > 0:
                        neurons = self.search_space['num_neurons'][idx - 1]
                    elif idx < len(self.search_space['num_neurons']) - 1:
                        neurons = self.search_space['num_neurons'][idx + 1]
            else:
                # With a small probability, activate the layer
                if random.random() < 0.1:
                    layer_exist = 1
                    neurons = random.choice(self.search_space['num_neurons'])
            new_layers.append((layer_exist, neurons))
        new_params['lstm_layers'] = new_layers

        # Ensure at least one LSTM layer is active
        if not any(layer_exist for layer_exist, _ in new_params['lstm_layers']):
            idx = random.randint(0, len(new_params['lstm_layers']) - 1)
            new_params['lstm_layers'][idx] = (1, random.choice(self.search_space['num_neurons']))

        # Perturb 'dropout_rate'
        idx = self.search_space['dropout_rate'].index(new_params['dropout_rate'])
        if random.random() < 0.5:
            if random.random() < 0.5 and idx > 0:
                new_params['dropout_rate'] = self.search_space['dropout_rate'][idx - 1]
            elif idx < len(self.search_space['dropout_rate']) - 1:
                new_params['dropout_rate'] = self.search_space['dropout_rate'][idx + 1]

        # Perturb 'optimizer' with a small probability
        if random.random() < 0.1:
            current_idx = self.search_space['optimizer'].index(new_params['optimizer'])
            new_idx = (current_idx + random.choice([-1, 1])) % len(self.search_space['optimizer'])
            new_params['optimizer'] = self.search_space['optimizer'][new_idx]

        # Perturb 'learning_rate'
        idx = self.search_space['learning_rate'].index(new_params['learning_rate'])
        if random.random() < 0.5:
            if random.random() < 0.5 and idx > 0:
                new_params['learning_rate'] = self.search_space['learning_rate'][idx - 1]
            elif idx < len(self.search_space['learning_rate']) - 1:
                new_params['learning_rate'] = self.search_space['learning_rate'][idx + 1]

        return new_params

    def train_and_evaluate(self, model, train_data, val_data):
        criterion = nn.MSELoss()
        model.train()
        epochs = 10  # Adjust as needed
        for epoch in range(epochs):
            for X_batch, y_batch in train_data:
                model.optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output.squeeze(), y_batch.squeeze())
                loss.backward()
                model.optimizer.step()
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_data:
                output = model(X)
                val_loss += criterion(output.squeeze(), y.squeeze()).item()
        return val_loss / len(val_data)

# Define the hyperparameter search space
search_space = {
    'num_neurons': list(range(1, 21)),  # Integers from 1 to 20
    'dropout_rate': [0.3, 0.4, 0.5, 0.6, 0.7],
    'optimizer': ['Adagrad', 'Adam', 'Adamax', 'RMSprop', 'SGD'],
    'learning_rate': [0.01, 0.001, 0.0001, 0.00001, 0.000001]
}

# Run ARO Optimization
aro = AROOptimizer(num_rabbits=10, max_iter=5, search_space=search_space)
best_params = aro.optimize(train_data, val_data, input_size=X_features.shape[2])

print("Best Parameters found by ARO:", best_params)

X_combined = torch.cat((X_train, X_val), dim=0)
y_combined = torch.cat((y_train, y_val), dim=0)
combined_data = DataLoader(TensorDataset(X_combined, y_combined), batch_size=32, shuffle=True)

# Build and Train Final Model with Best Parameters
best_model = LSTMModel(input_size=X_features.shape[2], layer_params=best_params)
criterion = nn.MSELoss()
epochs = 50  # Adjust as needed
for epoch in range(epochs):
    best_model.train()
    for X_batch, y_batch in combined_data:
        best_model.optimizer.zero_grad()
        output = best_model(X_batch)
        loss = criterion(output.squeeze(), y_batch.squeeze())
        loss.backward()
        best_model.optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}")

def make_predictions(model, test_data, use_predictions=False, window_size=5):
    model.eval()
    predictions = []
    actuals = []

    def debug_print_nans(array, name):
        if isinstance(array, (np.ndarray, pd.Series)):
            if np.any(np.isnan(array)):
                print(f"NaN found in {name}")
                print(f"Shape: {array.shape if isinstance(array, np.ndarray) else array.shape[0]}")
                print(f"NaN positions: {np.where(np.isnan(array))}")
        elif np.isnan(array):
            print(f"NaN found in {name} (scalar value)")

    # Calculate historical volatility and bounds
    historical_std = data['Close'].std()
    max_price_change = 2 * historical_std

    # Initialize historical data with more periods to handle technical indicators
    lookback = 50  # Enough periods for all indicators
    historical_closes = data['Close'].values[-lookback:].flatten().astype(float).tolist()
    last_actual_price = historical_closes[-1]
    prediction_window = []

    print(f"\nInitial setup with {lookback} periods of historical data")
    print(f"Historical closes length: {len(historical_closes)}")
    print(f"Last actual price: {last_actual_price}")

    with torch.no_grad():
        current_sequence = X_combined[-1].unsqueeze(0)
        
        for i in range(len(test_data.dataset)):
            # Get actual value and store it
            actual = y_test[i].numpy().item()
            actual_unscaled = price_scaler.inverse_transform([[actual]])[0][0]
            actuals.append(actual_unscaled)

            # Make prediction
            pred = model(current_sequence).numpy()
            debug_print_nans(pred, "model prediction")
            
            pred_price = float(pred[0][0])
            pred_price_unscaled = price_scaler.inverse_transform([[pred_price]])[0][0]
            
            # Apply bounds
            last_price = historical_closes[-1]
            debug_print_nans(np.array([last_price]), "last_price")
            
            max_change = max_price_change * (1.0 / (i + 1)**0.5)
            pred_price_unscaled = np.clip(pred_price_unscaled, 
                                        last_price - max_change,
                                        last_price + max_change)

            # Smooth prediction
            prediction_window.append(pred_price_unscaled)
            if len(prediction_window) > window_size:
                prediction_window.pop(0)
            smoothed_pred_unscaled = np.mean(prediction_window)
            
            # Store prediction
            predictions.append(smoothed_pred_unscaled)

            # Update historical prices
            if use_predictions:
                historical_closes.append(smoothed_pred_unscaled)
            else:
                historical_closes.append(actual_unscaled)
            historical_closes.pop(0)

            # Calculate technical indicators with default values for initialization period
            historical_closes_arr = np.array(historical_closes, dtype=float)
            
            # SMA with fallback
            sma_20 = ta.SMA(historical_closes_arr, timeperiod=20)
            sma_20 = float(sma_20[-1] if sma_20 is not None and not np.isnan(sma_20[-1]) 
                          else np.mean(historical_closes_arr[-20:]))
            
            # MACD with fallback
            macd, macd_signal, macd_hist = ta.MACD(historical_closes_arr)
            macd = float(macd[-1] if macd is not None and not np.isnan(macd[-1]) 
                        else historical_closes_arr[-1] - np.mean(historical_closes_arr[-26:]))
            macd_signal = float(macd_signal[-1] if macd_signal is not None and not np.isnan(macd_signal[-1]) 
                              else macd)
            macd_hist = float(macd_hist[-1] if macd_hist is not None and not np.isnan(macd_hist[-1]) 
                            else 0)
            
            # RSI with fallback
            rsi = ta.RSI(historical_closes_arr, timeperiod=14)
            rsi = float(rsi[-1] if rsi is not None and not np.isnan(rsi[-1]) else 50)
            
            # ATR with fallback
            highs = np.array([price * 1.001 for price in historical_closes_arr])
            lows = np.array([price * 0.999 for price in historical_closes_arr])
            atr = ta.ATR(highs, lows, historical_closes_arr, timeperiod=14)
            atr = float(atr[-1] if atr is not None and not np.isnan(atr[-1]) 
                       else np.std(historical_closes_arr[-14:]))
            
            # OBV with fallback
            obv = ta.OBV(historical_closes_arr, np.ones_like(historical_closes_arr))
            obv = float(obv[-1] if obv is not None and not np.isnan(obv[-1]) else 0)

            # Get market data
            market_idx = -len(test_data.dataset) + i
            market_close = float(market_data['Close'].iloc[market_idx])
            market_sma_20 = float(market_data['SMA_20'].iloc[market_idx])
            market_macd = float(market_data['MACD'].iloc[market_idx])
            market_macd_signal = float(market_data['MACD_signal'].iloc[market_idx])
            market_macd_hist = float(market_data['MACD_hist'].iloc[market_idx])
            market_rsi = float(market_data['RSI'].iloc[market_idx])
            market_atr = float(market_data['ATR'].iloc[market_idx])
            market_obv = float(market_data['OBV'].iloc[market_idx])

            # Create feature array with all features
            features_unscaled = np.array([[
                float(smoothed_pred_unscaled if use_predictions else actual_unscaled),
                sma_20,
                macd,
                macd_signal,
                macd_hist,
                rsi,
                atr,
                obv,
                market_close,
                market_sma_20,
                market_macd,
                market_macd_signal,
                market_macd_hist,
                market_rsi,
                market_atr,
                market_obv
            ]], dtype=np.float32)
            
            debug_print_nans(features_unscaled, "features_unscaled")
            
            # Scale features
            new_features = scaler.transform(features_unscaled)
            debug_print_nans(new_features, "new_features")

            # Update sequence
            if use_predictions:
                current_sequence = current_sequence[:, 1:, :]
                new_features_tensor = torch.FloatTensor(new_features).unsqueeze(0)
                current_sequence = torch.cat([
                    current_sequence,
                    new_features_tensor
                ], dim=1)
            else:
                if i < len(test_data.dataset) - 1:
                    next_actual = X_test[i+1]
                    current_sequence = torch.cat([
                        current_sequence[:, 1:, :],
                        next_actual[-1].unsqueeze(0).unsqueeze(0)
                    ], dim=1)

            if i == 0:  # Print full arrays for first iteration
                print("\nFirst iteration values:")
                print(f"Prediction: {pred_price_unscaled}")
                print(f"SMA_20: {sma_20}")
                print(f"MACD: {macd}")
                print(f"RSI: {rsi}")
                print(f"ATR: {atr}")
                print(f"Features unscaled shape: {features_unscaled.shape}")
                print(f"Features unscaled: {features_unscaled}")

    return np.array(predictions).reshape(-1, 1), np.array(actuals).reshape(-1, 1)


def calculate_direction_accuracy(actual_prices, pred_prices, prev_prices):
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

def evaluate_predictions(pred_prices, actual_prices, mode=""):
    mae = mean_absolute_error(actual_prices, pred_prices)
    mse = mean_squared_error(actual_prices, pred_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, pred_prices)
    
    # Calculate directional accuracy
    direction_metrics = calculate_direction_accuracy(actual_prices, pred_prices, None)

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

# Run both prediction methods with different initial conditions
predictions_actual, actuals = make_predictions(best_model, test_data, use_predictions=False)
predictions_pred, _ = make_predictions(best_model, test_data, use_predictions=True)

# No need for inverse transform since we're already working with unscaled prices
pred_prices_actual = predictions_actual
pred_prices_pred = predictions_pred
actual_prices = actuals

# Evaluate both methods
metrics_actual = evaluate_predictions(pred_prices_actual, actual_prices, "Using Actual Values")
metrics_pred = evaluate_predictions(pred_prices_pred, actual_prices, "Using Prior Predictions")

# Plot results
plot_predictions(pred_prices_actual, pred_prices_pred, actual_prices)

# Print sample of predictions with direction indicators
print("\nSample of Predictions:")
print("Day | Previous Close | Actual Price | Prediction (Actual) | Prediction (Prior) | Actual Move | Pred Move (A) | Pred Move (P)")
print("-" * 120)
for i in range(min(10, len(actual_prices))):
    prev_close = float(data['Close'].values[-(len(actual_prices)-i+1)] if i > 0 else data['Close'].values[-len(actual_prices)-1])
    
    # Calculate price movements
    actual_move = '↑' if actual_prices[i][0] > prev_close else '↓' if actual_prices[i][0] < prev_close else '→'
    pred_move_actual = '↑' if pred_prices_actual[i][0] > prev_close else '↓' if pred_prices_actual[i][0] < prev_close else '→'
    pred_move_prior = '↑' if pred_prices_pred[i][0] > prev_close else '↓' if pred_prices_pred[i][0] < prev_close else '→'
    
    print(f"{i+1:3d} | {prev_close:13.2f} | {actual_prices[i][0]:11.2f} | "
          f"{pred_prices_actual[i][0]:17.2f} | {pred_prices_pred[i][0]:17.2f} | "
          f"{actual_move:^11s} | {pred_move_actual:^11s} | {pred_move_prior:^11s}")

