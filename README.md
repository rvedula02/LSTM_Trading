# Optimizing Neural Networks for Medium-Frequency Trading in the American Stock Market

By Arnav Kolli, Arjun Hegde, Rahul Vedula

## Overview
This project investigates the application of Long Short-Term Memory (LSTM) neural networks to medium-frequency trading in the American stock market. It integrates technical indicators and employs the Adaptive Rabbit Optimization (ARO) algorithm to optimize hyperparameters. The study compares LSTM models with traditional statistical methods and alternative machine learning models, such as CNN and hybrid LSTM-CNN architectures, to highlight their advantages in capturing market dynamics and improving prediction accuracy.

## Key Features
1. **Objective**: Evaluate LSTM's ability to predict stock price movements based on temporal data.
2. **Data Sources**: Historical stock data from **yfinance** API, including technical indicators (SMA, EMA, RSI, MACD).
3. **Optimization**: Adaptive Rabbit Optimization (ARO) algorithm for hyperparameter tuning.
4. **Models**:
   - Baseline LSTM
   - LSTM with market-wide features
   - Pure CNN
   - Hybrid LSTM-CNN

## Experiments and Results
1. **Baseline LSTM**:
   - **MAE**: 4.95 | **R²**: 0.70 (actual values)
   - Accuracy deteriorates with multi-step predictions.
2. **LSTM with Market-Wide Features**:
   - **MAE**: 7.27 | **R²**: 0.23
   - Market-wide features added noise, reducing performance.
3. **CNN**:
   - **MAE**: 7.62 | **R²**: 0.18
   - Effective for short-term patterns but lacks sequential modeling capability.
4. **Hybrid LSTM-CNN**:
   - Best performer: **MAE**: 1.91 | **R²**: 0.93 (actual values)
   - Combines spatial and temporal feature extraction for robust predictions.

## Key Findings
- The hybrid LSTM-CNN model outperformed other architectures in accuracy and robustness.
- Challenges include handling sudden market shifts and cumulative prediction errors over time.
- LSTMs effectively capture temporal dependencies, showing potential for real-time market predictions.
