import pandas as pd
import datetime
import numpy as np
import scipy as sp
import os
import sys
sys.path.append('.')
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import StandardScaler
# https://github.com/civisanalytics/python-glmnet/issues/66
# brew install gcc@9 #  installed in /usr/local/Cellar/gcc@9/9.3.0/
# ln -s /usr/local/Cellar/gcc@9/9.3.0/lib/gcc/9 /usr/local/opt/gcc/lib/gcc/9
import glmnet_python
import glmnet
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPredict import cvglmnetPredict
from py.load_claim_data import load_claim_data
from py.load_5y_search_data import load_5y_search_data
from py.var_generator import var_generator

# data_5y = load_5y_search_data()
# data = data_5y['claim_data']
# data_early = data_5y['claim_earlyData']
# GTdata = data_5y['allSearch']
# decomp = "classic"
# n_history = 700
# n_training = 156
# alpha = 1
# UseGoogle = True
# nPred_vec = range(0, 4)
# discount = 0.015
# sepL1 = False

def prism(data, data_early, GTdata, decomp="classic", n_history = 700, n_training = 156, alpha = 1,
                UseGoogle = True, nPred_vec=range(0,3), discount = 0.015, sepL1 = False, seed=42):

    # cannot change s.window
    var = var_generator(data = data, data_early = data_early, decomp = decomp)

    start_id = len(data)
    end_id = len(data)

    # n.lags determined by var
    n_lag = list(range(0,var['y_lags'].shape[1]))

    lasso_pred = np.zeros((len(data), max(nPred_vec)+1))
    lasso_pred_cols = ["forcast_"+str(i) for i in nPred_vec]
    lasso_coef = {}

    for nPred in nPred_vec:
        if len(GTdata) > 0  and UseGoogle:
            lasso_coef[nPred+1] = np.zeros((len(data), GTdata.shape[1] + len(n_lag) + 1))
        else:
            lasso_coef[nPred+1] = np.zeros((len(data), len(n_lag) + 1))


    for nPred in nPred_vec:
        if len(GTdata) > 0  and UseGoogle:
            if not np.all(data.index==GTdata.index):
                raise ValueError("error in data and GTdata: their time steps (index) must match")
            penalty_factor = list(np.ones(np.max(n_lag) + 1)) + list(np.ones(GTdata.shape[1]))
            penalty_factor = np.array(penalty_factor).astype(sp.float64)
            design_matrix_all = np.column_stack((var["y_lags"][:, n_lag],GTdata.values))
        else:
            penalty_factor = np.ones(np.max(n_lag) + 1).astype(sp.float64)
            design_matrix_all = var["y_lags"][:,n_lag]

        # date[current+1] is the date
        for current in range(start_id - 1, end_id):
            training_idx = np.arange(current - n_training + nPred + 1, current + 1)
            # glmnet require y to be scipy float 64 type
            y_response = data.iloc[training_idx,:].values.squeeze().astype(sp.float64)

            design_matrix_all_to_scale = design_matrix_all[training_idx - nPred - 1, :]
            # https://stackoverflow.com/a/27297618
            sc = StandardScaler(with_mean=True, with_std=True)
            sc.fit(design_matrix_all_to_scale)
            sc.std_ = np.std(design_matrix_all_to_scale, axis=0, ddof=1)
            design_matrix = sc.transform(design_matrix_all_to_scale)

            # standardize covariates for data available on date[current+1]
            newx = sc.transform(design_matrix_all[current,:].reshape(1, -1))

            weights = pow(1-discount,np.arange(len(training_idx),0,-1))
            weights = np.array(weights).astype(sp.float64)

            if sepL1:
                pass
            else:
                # fit model
                if alpha is not None:
                    # edit /Users/mm28542/.virtualenvs/prism/lib/python3.8/site-packages/glmnet_python/cvglmnet.py line 260:
                    #ma = scipy.tile(scipy.arange(nfolds), [1, int(scipy.floor(nobs/nfolds))])

                    # https://github.com/bbalasub1/glmnet_python/issues/17
                    # https://github.com/bbalasub1/glmnet_python/files/2027121/GLMnet.so.zip
                    # cp ~/Download/GLMnet.so /Users/mm28542/.virtualenvs/prism/lib/python3.8/site-packages/glmnet_python/
                    # sudo cp /Library/Frameworks/R.framework/Versions/3.6/Resources/lib/libgfortran.3.dylib /usr/local/gfortran/lib/

                    # glmnet_python does not support random_state argument, so for random cv, we need to
                    # manually shuffle rows.
                    np.random.seed(seed)
                    idx = np.random.permutation(design_matrix.shape[0])

                    lasso_fit = cvglmnet(x=design_matrix[idx,:],
                                         y=y_response[idx],
                                         weights=weights, nfolds=10, grouped=False,
                                         alpha=alpha, penalty_factor = penalty_factor)

                else:
                    pass

                # save coef and predict
                if alpha is not None:
                    lasso_coef[nPred+1][current, :] = cvglmnetCoef(lasso_fit, s = 'lambda_1se').transpose()
                    lasso_pred[current, nPred] = cvglmnetPredict(lasso_fit, newx = newx, s='lambda_1se')
                else:
                    pass


        # colnames(lasso.coef[[nPred + 1]]) = rownames(as.matrix(coef(lasso.fit, lambda = lasso.fit$ lambda .1se)))
        lasso_coef[nPred+1] = lasso_coef[nPred+1][range(np.max([start_id, n_training + nPred]), end_id), :]

    xts_pred = lasso_pred[range(start_id - 1, end_id),:]

    return {'pred':xts_pred, 'pred_date': data.index.values[range(start_id - 1, end_id)], 'coef': lasso_coef}



if __name__ == "__main__":
    data_5y = load_5y_search_data(folder='0408')
    data = data_5y['claim_data']#.iloc[:200,:]
    #data.iloc[199,:] = None
    data_early = data_5y['claim_earlyData']
    var_gen = var_generator(data, data_early, col='icnsa', decomp="classic")
    GTcolsR = ["california unemployment", "claim unemployment", "file unemployment", "indiana unemployment",
              "new york unemployment", "nj unemployment", "ny unemployment", "nys unemployment", "ohio unemployment",
              "pa unemployment", "texas unemployment","unemployed","unemployment benefits","unemployment extension",
              "unemployment florida", "unemployment insurance",  "unemployment login", "unemployment number",
              "unemployment office","unemployment online","unemployment oregon","unemployment rate",
              "unemployment washington", "unemployment wisconsin",  "unemployment"]
    GTdata = data_5y['allSearch'][GTcolsR]#.iloc[:200,:]
    decomp = "classic"
    n_history = 700
    n_training = 156
    alpha = 1
    UseGoogle = True
    nPred_vec = range(0, 4)
    discount = 0.015
    sepL1 = False

    results = prism(data=data, data_early=data_early, GTdata=GTdata,
                    decomp=decomp, n_history=n_history, n_training=n_training, alpha=alpha,
          UseGoogle=UseGoogle, nPred_vec=nPred_vec, discount=discount, sepL1=sepL1, seed=1234)

    print(results['pred'])


