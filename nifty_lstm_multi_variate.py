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
from keras.layers import Dropout
import talib as ta
import tensorflow as tf
import random
from keras.callbacks import EarlyStopping
import pickle

def std_dev(data):
    # Get number of observations
    n = len(data)
    # Calculate mean
    mean = sum(data) / n
    # Calculate deviations from the mean
    deviations = sum([(x - mean)**2 for x in data])
    # Calculate Variance & Standard Deviation
    variance = deviations / (n - 1)
    s = variance**(1/2)
    return s

# Sharpe Ratio From Scratch
def sharpe_ratio(data, risk_free_rate=0.0):
    # Calculate Average Daily Return
    mean_daily_return = sum(data) / len(data)
    # Calculate Standard Deviation
    s = std_dev(data)
    # Calculate Daily Sharpe Ratio
    daily_sharpe_ratio = (mean_daily_return - risk_free_rate) / s
    # Annualize Daily Sharpe Ratio
    sharpe_ratio = 252**(1/2) * daily_sharpe_ratio    
    return sharpe_ratio

# Step 1: Download historical stock price data
def download_stock_data(symbol):
    data = yf.download(symbol, period='10y')
    return data

# Step 1: Download historical stock price data
def add_features(data):
    features_list = []
    data['upper_band'], data['middle_band'], data['lower_band'] = ta.BBANDS(data['Close'].values)
    data['macd'], data['macdsignal'], data['macdhist'] = ta.MACD(data['Close'].values)
    data['rsi'] = ta.RSI(data['Close'].values)
    #data['sar'] = ta.SAR(data['High'].values, data['Low'].values)
    data ['sma'] = data['Close'].rolling(window=20).mean()    
    data['lma'] = data['Close'].rolling(window=50).mean()
    data['obv'] = ta.OBV(data['Close'], data['Volume'])
    data['mfi'] = ta.MFI(data['High'], data['Low'], data['Close'], data['Volume'])
    features_list +=['upper_band','middle_band','lower_band','macd','rsi','sma','lma','obv','mfi']
    data = data.drop(['Adj Close', 'Open','High','Low','Volume'], axis=1) 
    return data,features_list

# Step 2: Data preprocessing
def preprocess_data(data):
    # Extract the 'Close' price is our target variable
    #df = data[['Close']]   
    df = data
    df.dropna(inplace=True)    
    # Normalize the data to a range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = scaler.fit_transform(df.values)
    out_scaler = MinMaxScaler(feature_range=(0, 1))
    temp = out_scaler.fit_transform(data['Close'].values.reshape(-1,1))    
    return df, scaler,out_scaler

# Step 3: Create sequences for the LSTM
def create_sequences(data, sequence_length):
    sequences = []
    target = []
    for i in range(len(data) - sequence_length):
        x = data[i:i+sequence_length,0:12]
        y = data[i+sequence_length,0]
        sequences.append(x)
        target.append(y)
    return np.array(sequences), np.array(target)

# Step 4: Build and train the LSTM model test  
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=100, activation='tanh',return_sequences=True, input_shape=input_shape))
    #model.add(Dropout(0.2))
    model.add(LSTM(units=100, activation='tanh',return_sequences=False))
    #model.add(Dropout(0.2))
    model.add(Dense(50,activation='tanh'))
    model.add(Dense(25,activation='tanh'))
    #model.add(Dropout(0.2))
    model.add(Dense(1,activation = 'linear'))
    #model.add(Dropout(0.2))
    model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy'])
    return model

def backtest(df, model):
    # Copy data
    data = df.copy()
    
    # Creating features
    features_list = []
    #data = data.drop(['Adj Close', 'Open','High','Low'], axis=1)    
    data,features_list = add_features(data)
    # Preprocess the data
    df,scaler,out_scaler = preprocess_data(data)
    
    # Define the sequence length and create sequences
    sequence_length = 10
    sequences, target = create_sequences(df, sequence_length)
    data['predicted']= np.nan    
    # Predict
    y_pred = model.predict(sequences)  
    y_pred = out_scaler.inverse_transform(y_pred.reshape(-1,1))
    result = pd.DataFrame(y_pred, columns = ['Predicted'])
    #= res
    data['predicted'][10:] = result['Predicted']
    # Create returns
    data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
    data['p_returns'] = data['Close'].pct_change()
    data['signal'] = np.where(data['predicted'] > data['Close']*1.01,1,0)    
    # Create strategy returns
    data['strategy_returns'] = data['returns'].shift(-1) * data['signal']
    data['p_strategy_returns'] = data['p_returns'].shift(-1) * data['signal']
    
   
    
    # Return the last cumulative strategy return
    # we need to drop the last nan value
    data.dropna(inplace=True)
    # Return the last cumulative return
    #bnh_returns = data['returns'].cumsum()[-1]
    #strategy_returns = data['strategy_returns'].cumsum()[-1]
    
    bnh_returns = (data['p_returns']+1).cumprod()[-1]
    strategy_returns = (data['p_strategy_returns']+1).cumprod()[-1]
    
    plt.figure(figsize=(10, 6))
    plt.plot(data['returns'].cumsum())
    plt.plot(data['strategy_returns'].cumsum())
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.title('Returns Comparison')
    plt.legend(["Buy and Hold Returns","Strategy Returns"])
    plt.show()
    
    return bnh_returns, strategy_returns, data


def train_and_save_model(symbol,train_datafile):
    random.seed(1)
    np.random.seed(1)
    tf.random.set_seed(1)
    data = pd.read_csv(train_datafile,index_col=0,parse_dates=True)
    # Creating features
    #data = data.drop(['Adj Close', 'Open','High','Low'], axis=1)    
    data,features_list = add_features(data)
    # Preprocess the data
    df,scaler,out_scaler = preprocess_data(data)
    
    # Define the sequence length and create sequences
    sequence_length = 10
    sequences, target = create_sequences(df, sequence_length)
    
    # Split the data into training and testing sets
    train_size = int(len(sequences) * 0.8)
    X_train, X_test = sequences[:train_size], sequences[train_size:]
    y_train, y_test = target[:train_size], target[train_size:]
    
    #print (X_train)
    #print(X_train.summary())
    # Build the LSTM model
    #Shape[1] is samples i.e. elements in sequencies and shape[2] is count of features
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(input_shape)
    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=20)
    
    # Train the model
    #model.fit(X_train, y_train, epochs=5, batch_size=10)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=1, callbacks=[es],batch_size=10)
    #history = model.fit(X_train, y_train,validation_data=(X_test, y_test), epochs=100, batch_size=10)
    plt.plot(range(len(model.history.history['loss'])), model.history.history['loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()
    
    #Evaluating model
    train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('Train: %.5f, Test: %.5f' % (train_acc[0], test_acc[0]))
    print('Train: %.5f, Test: %.5f' % (train_acc[1], test_acc[1]))
    
    # plot training history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    #y_pred = y_pred.shift(-1)
    
    # Inverse transform the predictions to get actual stock prices
    y_pred = out_scaler.inverse_transform(y_pred.reshape(-1,1))    
    #print (y_test)
    y_test = out_scaler.inverse_transform(y_test.reshape(-1, 1))
    print('After: ')
    #print(y_test)
    
    # Calculate and print the Mean Squared Error (MSE)
    #mse = mean_squared_error(y_test, y_pred)
    #print(f'Mean Squared Error: {mse}')
    
    result = pd.DataFrame(y_test, columns = ['Actual'])
    result2 = pd.DataFrame(y_pred, columns = ['Predicted'])
    result2 = result2.shift(1)
    final_result = pd.concat([result,result2], axis=1)
    #print(final_result)
    final_result.dropna(inplace=True)
    
    model_performance(final_result['Actual'],final_result['Predicted']) 
    
    final_result['diff'] = final_result['Actual'] - final_result['Predicted']
    final_result['PC'] = np.where ( ( (final_result['Actual'] >  final_result['Actual'].shift(1)) & (final_result['Predicted'] >  final_result['Actual'].shift(1) )),1,0)
    final_result['PC'] = np.where ( ( (final_result['Actual'] <  final_result['Actual'].shift(1)) & (final_result['Predicted'] <  final_result['Actual'].shift(1) )),1,final_result['PC'])
    print(final_result['PC'].value_counts())
    
    # Plot the actual vs. predicted stock prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()
    
    pickle.dump(model, open('model.pkl', 'wb'))
    

# Step 5: Main function for stock price prediction
def main():
    # Define the stock symbol, start date, and end date
    symbol = '^NSEI'
    train_file_name = 'nifty_18_09.csv'    
    train_and_save_model(symbol,train_file_name) 
    
    model = pickle.load(open('model.pkl', 'rb'))
    
    #data2 = pd.read_csv("nifty_18_09.csv",index_col=0,parse_dates=True)
    data2 = download_stock_data('^NSEI')
    bnh_returns, s_returns, data = backtest(data2,model)
    print ('Start Value', data['Close'][0])
    print ('End Value ', data['Close'][-1])
    print('Buy and Hold Returns:', bnh_returns)
    print('Strategy Returns:', s_returns)
    # Get the summary statistics for the strategy using pyfolio
    print(data)
    import pyfolio as pf

    pf.create_full_tear_sheet(data['strategy_returns'])

if __name__ == "__main__":
    main()
