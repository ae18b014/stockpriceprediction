"""
Statistical Feature Extraction from Stock df.

Data used:  GOOGL stock data from '2010-06-21' - '2020-06-12'
            Correlated Assets of JPM, MS, NYSE, BSE, HANGSENG, USD_GBP, LIBOR_USD, VIX,	NIKKEI, NASDAQ -> INCLUDED

@author: aayshaanserbabu
"""

import pandas as pd
import numpy as np
import datetime

def get_technical_indicators(df):
    """
       Technical Indicators
    --------------------------------
        Moving Average:
        --------------
                Takes Average of the past few days (window)

        Exponential Moving Average:
        --------------------------
                More sensitive than simple MA, gives more weight to recent days than older

         Moving Average Convergense Divergense:
         -----------------------------------
                EMA(12 DAYS) - EMA(26 DAYS)

        Bollinger Bands:
        ----------------
                MA(TP,n) +- m*(std(TP,n))
                {m : no. standard deviation
                TP: Trading Price
                n : no. of days in the window}
    """
    #adding moving averages 7, 21 days
    df['ma7'] = df['Close'].rolling(7, min_periods = 1).mean()
    df['ma21'] = df['Close'].rolling(21, min_periods = 1).mean()
    #exponential ma 12,26,days, and calculating MACD
    df['26ema'] = df['Close'].ewm(span = 26).mean()
    df['12ema'] =df['Close'].ewm(span = 12).mean()
    df['MACD']  = df['12ema'] - df['26ema']
    #bollinger bamds
    df['20sd']  = df['Close'].rolling(window =20, min_periods = 1).std()
    
    df['upper_band'] = df['ma21'] + df['20sd']
    df['lower_band'] = df['ma21'] - df['20sd']
    #exponential ma
    df['ema'] = df['Close'].ewm(com = 0.5).mean()
    #momentum 
    df['momentum'] = df['Close'] - 1
    df['log_momentum'] = df['momentum'].apply(np.log)
    
    df.drop(index = 0,inplace = True)
    
    return df

def get_fourier_transforms(df):
    """
    Return Fourier Transforms of the closing price
    taking 3, 6, 9, 100 components at a time
    """
    close_fft = np.fft.fft(np.asarray(df['Close'].tolist()))
    fft_list = np.asarray(close_fft.tolist())
    for num_ in [3, 6, 9, 100]:
        fft_list_m10= np.copy(fft_list)
        fft_list_m10[num_:-num_]=0
        inv = np.fft.ifft(fft_list_m10)
        df['FFT_' + str(num_) + '_abs']  = np.abs(inv)
        df['FFT_' + str(num_) + '_angle']= np.angle(inv)
    return df


if __name__ =='__main__':
    data = pd.read_csv('GOOGL_stock_data.csv', parse_dates=['Date'])
    print(data.head())
    data = get_technical_indicators(data)
    data = get_fourier_transforms(data)
    print(data.head())




