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
                  n_lag = range(0,52), s_window = 52, n_history = 700):
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
    var_gen = {"y_lags": y_lags}
    return var_gen

if __name__ == "__main__":
    data_5y = load_5y_search_data()
    data = data_5y['claim_data']
    data_early = data_5y['claim_earlyData']
    var_gen = var_generator(data, data_early, col='icnsa', decomp="classic")

