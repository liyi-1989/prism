import pandas as pd
import datetime
import numpy as np
import os
import sys
sys.path.append('.')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from py.load_claim_data import load_claim_data
from py.load_5y_search_data import load_5y_search_data

def var_generator(data, data_early, col='icnsa', decomp = "classic",
                  n_lag = range(1,53), s_window = 52, n_history = 700):
    """
    :param data: data used to generate features
    :param data_early: previous data used to do the time series decomposition
    :param col: column name used in data
    :param decomp: decomposition method (classic or stl)
    :param n_lag: how many historical days used as features
    :param s_window: seasonal window, used in the seasonal decomposition freq/period parameter
    :param n_history: (for a given time point, its) past time series length used to derive the time series decomposition features
    :return:
    y_lags: numpy array,
    y_lags_cols: column names (row names is data.index)
    """
    n_past = len(data_early) # the length of all historical data
    data_comb = data_early.append(data)
    y_lags = np.zeros((len(data), 2 * len(n_lag))) # ntimes * (lag of z, lag of s)

    for i in range(0, len(data)):
        y_vec = data_comb[(i+n_past-n_history):(i+n_past)][col]
        if decomp == "stl":
            pass
        elif decomp == "classic":
            decom = seasonal_decompose(y_vec,model='additive',period=s_window)
            y_trend = y_vec - decom.seasonal
            y_lags[i,:] = np.append(y_trend[-len(n_lag):].values,
                                    decom.seasonal[-len(n_lag):].values)
        else:
            pass
    colsz = ['z_lag_' + str(i) for i in n_lag]
    colss = ['s_lag_' + str(i) for i in n_lag]
    colsz.reverse()
    colss.reverse()
    y_lags_cols = colsz + colss
    var_gen = {"y_lags": y_lags,
               "y_lags_cols": y_lags_cols
               }
    return var_gen

if __name__ == "__main__":
    data_5y = load_5y_search_data()
    data = data_5y['claim_data']
    data_early = data_5y['claim_earlyData']
    var_gen = var_generator(data, data_early, col='icnsa', decomp="classic")

