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
from py.var_generator import var_generator


# def prism(data, data_early, GTdata, decomp="classic", n_history = 700, n_training = 156, alpha = 1,
#                 UseGoogle = True, nPred_vec=range(0,3), discount = 0.015, sepL1 = False):
#

data_5y = load_5y_search_data()
data = data_5y['claim_data']
data_early = data_5y['claim_earlyData']
GTdata = data_5y['allSearch']
decomp="classic"
n_history = 700
n_training = 156
alpha = 1
UseGoogle = True
nPred_vec=range(0,3)
discount = 0.015
sepL1 = False

# cannot change s.window
var = var_generator(data = data, data_early = data_early, decomp = decomp)



if __name__ == "__main__":
    data_5y = load_5y_search_data()
    data = data_5y['claim_data']
    data_early = data_5y['claim_earlyData']
    var_gen = var_generator(data, data_early, col='icnsa', decomp="classic")


