# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 12:50:20 2023

@author: AshishJain
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf  # You can install this package using: pip install yfinance
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime
from tsa_functions_quantra import model_performance

# Step 1: Download historical stock price data
def download_stock_data(symbol, start_date, end_date):
    data = yf.download(symbol, period='10y')
    return data

# Step 2: Data preprocessing
def preprocess_data(data):
    # Extract the 'Close' price as our target variable
    df = data[['Close']]
    
    # Normalize the data to a range between 0 and 1
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    
    return df, scaler

# Step 3: Create sequences for LSTM
def create_sequences(data, sequence_length):
    sequences = []
    target = []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length]
        y = data[i+sequence_length]
        sequences.append(x)
        target.append(y)
    return np.array(sequences), np.array(target)

# Step 4: Build and train the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Main function for stock price prediction
def main():
    # Define the stock symbol, start date, and end date
    symbol = '^NSEI'
    start_date = '2015-01-01'
    end_date = datetime.date.today()
    #end_date = '2023-01-01'
    
    # Download historical stock data
    data = download_stock_data(symbol, start_date, end_date)
    
    # Preprocess the data
    df, scaler = preprocess_data(data)
    
    # Define the sequence length and create sequences
    sequence_length = 60
    sequences, target = create_sequences(df.values, sequence_length)
    
    # Split the data into training and testing sets
    train_size = int(len(sequences) * 0.8)
    X_train, X_test = sequences[:train_size], sequences[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    
    print (X_train)
    # Build the LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # Train the model
    model.fit(X_train, y_train, epochs=3, batch_size=1)
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Inverse transform the predictions to get actual stock prices
    y_pred = scaler.inverse_transform(y_pred)    
    #print (y_test)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    print('After: ')
    #print(y_test)
    
    # Calculate and print the Mean Squared Error (MSE)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    rmse = np.sqrt(np.mean(y_pred - y_test)**2)
    print(f'Root Mean Squared Error: {rmse}')
    result = pd.DataFrame(y_test, columns = ['Actual'])
    result2 = pd.DataFrame(y_pred, columns = ['Predicted'])
    final_result = pd.concat([result,result2], axis=1)
    print(final_result)
    
    model_performance(final_result['Actual'],final_result['Predicted']) 
    
    # Plot the actual vs. predicted stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(data.index[train_size+sequence_length:], y_test, label='Actual Price')
    plt.plot(data.index[train_size+sequence_length:], y_pred, label='Predicted Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
