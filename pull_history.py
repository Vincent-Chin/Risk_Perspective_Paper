# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 19:46:07 2015

@author: vincent

These methods pull data for a given period for a given instrument type,
and return dataframes.  they're meant to be stored in our database;
these methods are meant to be overridden by higher-quality data-source
methods later on.
"""

from pandas_datareader import data
import numpy as np
import pandas as pd
import time


# constants
TIINGO_API_Key = ""  # scrubbed.  replace with your own:  https://api.tiingo.com/


#######################################################################
def getEquity(symbol, initDate, finalDate, method='tiingo', exWeekends=True):
    startDate = np.datetime64(initDate, 'D')
    endDate = np.datetime64(finalDate, 'D')

    retryLimit = 0
    info = pd.DataFrame()
    while retryLimit < 5:
        try:
            access_key = None
            if method == 'tiingo':
                access_key = TIINGO_API_Key
                if np.busday_count(startDate, endDate) == 0:
                    return info
                else:
                    info = data.DataReader(symbol, method, startDate, endDate, access_key=access_key)
                    info = Format_Tiingo_ToDBCompatible(info)
            else:
                info = data.DataReader(symbol, method, startDate, endDate)

        except Exception as e:
            print("Exception during getEquity on", symbol, retryLimit,
                  "\nMessage:", repr(e))

        if 'Open' in info.columns and len(info) > 0:
            info.sort_index(inplace=True)
            info = dropDirty(info, exWeekends)
            break  # if no exception, no point in retrying
        else:
            retryLimit += 1

        # sleep a second for the next retry.
        time.sleep(1)

    if retryLimit >= 5:
        print("getEquity: Hit retry limit for", symbol)

    return info


#######################################################################
def Format_Tiingo_ToDBCompatible(info):
    baseData = info.copy().reset_index()
    baseData['Timestamp'] = baseData.date
    baseData['Open'] = baseData.open
    baseData['High'] = baseData.high
    baseData['Low'] = baseData.low
    baseData['Close'] = baseData.close
    baseData['AdjClose'] = baseData.adjClose
    baseData['Volume'] = baseData.adjVolume
    baseData = baseData.set_index('Timestamp')
    baseData = baseData[['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']]
    return baseData


#######################################################################


#######################################################################
def dropDirty(history, exWeekends):
    history = history[(history.Open != 0)
                    & (history.High != 0)
                    & (history.Low != 0)
                    & (history.Close != 0)]

    history = history[(pd.isnull(history.Open) == False)
                    & (pd.isnull(history.High) == False)
                    & (pd.isnull(history.Low) == False)
                    & (pd.isnull(history.Close) == False)]

    # we're going to drop any days where the open and high and low and close
    # equal one another.  these are invalid (closed) days
    history = history[((history.Open == history.Close)
                      & (history.Open == history.High)
                      & (history.Open == history.Low)) == False]

    if exWeekends:
        dts = pd.to_datetime(history.index).weekday
        history = history[dts < 5] # cut out Saturday[5] and Sunday[6]

    return history
#######################################################################
