# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 22:01:42 2023

@author: AshishJain
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#%matplotlib inline
plt.style.use('seaborn-darkgrid')
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)

import itertools
from tqdm import tqdm
# Data manipulation
import numpy as np
import pandas as pd
import datetime
import yfinance as yf

# For statistical analysis
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf

# Import adfuller
from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima.model.ARIMA',
                        FutureWarning)
from tsa_functions_quantra import model_performance

def check_stationarity(df):
    adf_result = adfuller(df)

    if(adf_result[1] < 0.05):
        return True
    else:
        return False
    
# Function to predict the price of 1 day
def predict_price_ARIMA(train_data,p,d,q):
    # Define model
    model = ARIMA(train_data, order=(p,d,q))
    # Fit the model
    #model.initialize_approximate_diffuse() # this line
    #model_fit = model.fit(start_params=model_fit_0.params)
    model_fit = model.fit()
    # Make forecast
    #print (model_fit.forecast())
    return model_fit.forecast()    

num_years = 10
warnings.filterwarnings('ignore')
end1 = datetime.date.today()
start1 = end1 - pd.Timedelta(days=num_years*365)
#start1 = datetime.datetime(2020,4,22)
#data = yf.download("^NSEI", start=start1, end=end1)
data = pd.read_csv("nifty_18_09.csv",index_col=0,parse_dates=True)
pd.get_option("display.max_columns")
pd.set_option('expand_frame_repr', False)

print(check_stationarity(data['Close']))
data.dropna(inplace=True)
print(check_stationarity(data['Close'].diff().dropna()))


# Drop the missing values
#data = data.dropna()

# Rolling Window
rolling_window = int(len(data)*0.70)

# Find the order of AR and MA
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

# Plot Partial Autocorrelation Function
plot_pacf(data['Close'][:rolling_window].diff().dropna(), lags=15, ax=ax1)
ax1.set_xlabel('Lags')
ax1.set_ylabel('Partial Autocorrelation')

# Plot Autocorrelation Function
plot_acf(data['Close'][:rolling_window].diff().dropna(), lags=15, ax=ax2)
ax2.set_xlabel('Lags')
ax2.set_ylabel('Autocorrelation')

plt.tight_layout()
plt.show()


# Import ARIMA and train model using training data with finalized order params 


split = int(len(data)*0.7)
data_train = data[:split]
data_test = data[split:]

# Train autoregressive model of order 35 and 37
model = ARIMA(data['Close'][:rolling_window], order = (6, 1, 2))
model_fit_0 = model.fit()
print(model_fit_0.params.round(2))


#%%timeit
# Predict the price using 'predict_price_ARIMA' function for Training Data
data['predicted_close'] = data['Close'].rolling(split).apply(predict_price_ARIMA,args =(6,1,2))
# Shift the predicted price by 1 period
#data_train['predicted_returns'] = data_train['predicted_returns'].shift(1)

print (data)


data['PC_SHIFTED'] = data['predicted_close'].shift(1)
data['diff'] = data['Close'] - data['predicted_close']
data['PC'] = np.where ( ( (data['Close'] >  data['Close'].shift(1)) & (data['predicted_close'] >  data['Close'].shift(1) )),1,0)
data['PC'] = np.where ( ( (data['Close'] <  data['Close'].shift(1)) & (data['predicted_close'] <  data['Close'].shift(1) )),1,data['PC'])
print(data['PC'].value_counts())
data['diff_shift'] = data['Close'] - data['PC_SHIFTED']
data['PCS'] = np.where (( (data['Close'] >  data['Close'].shift(1)) & (data['PC_SHIFTED'] >  data['Close'].shift(1) )),1,0)
data['PCS'] = np.where (( (data['Close'] <  data['Close'].shift(1)) & (data['PC_SHIFTED'] <  data['Close'].shift(1) )),1,data['PCS'])
print(data['PCS'].value_counts())
print(data)

data.dropna(inplace=True)

model_performance(data['Close'],data['predicted_close']) 
model_performance(data['Close'],data['PC_SHIFTED']) 


